import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration with modern defaults"""
    vocab_size: int = None
    context_length: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout_rate: float = 0.1
    bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 50
    compile_model: bool = False
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads

class RMSNorm(nn.Module):
    """Root Mean Square Normalization - more efficient than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.weight * x

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings for better length generalization"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
        
    def _update_cache(self, seq_len: int, device):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()[None, :, None, :]
            self._cached_sin = emb.sin()[None, :, None, :]
            
    def forward(self, q, k):
        batch, seq_len, n_heads, d_head = q.shape
        self._update_cache(seq_len, q.device)
        cos = self._cached_cos[:, :seq_len, :, :d_head]
        sin = self._cached_sin[:, :seq_len, :, :d_head]
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot
    
    def _apply_rotation(self, x, cos, sin):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        ], dim=-1)

class SwiGLU(nn.Module):
    """SwiGLU activation - shown to improve performance over ReLU"""
    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.up_proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.down_proj = nn.Linear(dim_out, dim_in, bias=bias)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = F.silu(gate)
        x = gate * up
        x = self.down_proj(x)
        return x

class EnhancedAttention(nn.Module):
    """Multi-head attention with Rotary embeddings and optimizations"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.rope = RotaryEmbedding(config.d_head, config.context_length, config.rope_theta)
        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.resid_dropout = nn.Dropout(config.dropout_rate)
        self.register_buffer('causal_mask', 
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1).bool())
        
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.config.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out

class TransformerBlock(nn.Module):
    """Enhanced transformer block with RMSNorm and SwiGLU"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = EnhancedAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff, bias=config.bias)
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer Language Model with modern improvements"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            torch.nn.init.normal_(block.attention.out_proj.weight, mean=0.0, 
                                 std=config.init_std / math.sqrt(2 * config.n_layers))
            torch.nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0,
                                 std=config.init_std / math.sqrt(2 * config.n_layers))
        print(f"Model initialized with {self.count_parameters():.2f}M parameters")
        
    def _init_weights(self, module):
        """Custom weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            
    def count_parameters(self):
        """Count trainable parameters- in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.config.context_length, f"Sequence length {T} exceeds maximum context length {self.config.context_length}"
        x = self.token_embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate tokens with temperature and top-k sampling"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# Streamlit
@st.cache_resource
def load_model():
    """Load the pretrained model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load('weights.pt', map_location=device, weights_only=False)
    
    # Extract components
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # Create model and load weights
    model = EnhancedTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create encode/decode functions
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    encode = lambda s: [char_to_idx.get(c, 0) for c in s]  # Use 0 for unknown chars
    decode = lambda l: ''.join([idx_to_char[i] for i in l])
    
    return model, encode, decode, device

# Streamlit UI
st.title("Shakespeare Text Generator")
st.title("Enter some text and let it continue in Shakespeare's style!")

# Load model
model, encode, decode, device = load_model()

# Input section
input_text = st.text_area("Enter your text:", height=100, 
                          placeholder="To be or not to be...")

# Generation parameters
col1, col2, col3 = st.columns(3)
with col1:
    max_tokens = st.slider("Max tokens", 50, 500, 200)
with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
with col3:
    top_k = st.slider("Top-k", 1, 100, 50)

# Generate button
if st.button("Generate", type="primary"):
    if input_text:
        with st.spinner("Generating..."):
            # Encode input
            encoded_input = encode(input_text)
            input_tensor = torch.tensor([encoded_input], dtype=torch.long, device=device)
            
            # Generate
            generated = model.generate(input_tensor, max_new_tokens=max_tokens, 
                                      temperature=temperature, top_k=top_k)
            
            # Decode and display
            output_text = decode(generated[0].tolist())
            
            st.markdown("Generated Text:")
            st.text_area("", output_text, height=300)
    else:
        st.warning("Enter some text first")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. Enter text in the input box
    2. Adjust the generation parameters:
       - **Max tokens**: How many characters to generate
       - **Temperature**: Higher = more creative, Lower = less creative
       - **Top-k**: Limits vocabulary to top k most likely tokens
    3. Click 'Generate' to continue your text in Shakespeare's style!

    """)
