
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "sslm"

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Model architecture
# Must match the training notebook
# --------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        y = x32 * torch.rsqrt(
            x32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (y * self.weight.float()).to(dtype=x.dtype)


class DiagonalLTI_SSM(nn.Module):
    def __init__(self, d_inner, d_state, max_len):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state
        self.max_len = max_len

        self.log_rates = nn.Parameter(
            torch.empty(d_inner, d_state).normal_(-1.5, 0.5)
        )
        self.B = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_inner))

        self.register_buffer(
            "time",
            torch.arange(max_len, dtype=torch.float32),
            persistent=False
        )

    def forward(self, u):
        original_dtype = u.dtype
        batch, length, channels = u.shape

        if length > self.max_len:
            raise ValueError("Sequence exceeds maximum length.")

        with torch.autocast(device_type="cpu", enabled=False):
            u32 = u.float().transpose(1, 2)

            rates = F.softplus(self.log_rates.float()) + 1e-4
            t = self.time[:length]

            decay = torch.exp(
                -rates.unsqueeze(-1) * t.view(1, 1, -1)
            )

            kernel = torch.einsum(
                "cn,cnl->cl",
                self.B.float() * self.C.float(),
                decay
            )

            n_fft = 2 * length

            u_f = torch.fft.rfft(u32, n=n_fft)
            k_f = torch.fft.rfft(kernel.float(), n=n_fft)

            y = torch.fft.irfft(
                u_f * k_f.unsqueeze(0),
                n=n_fft
            )[..., :length]

            y = y + u32 * self.D.float().view(1, -1, 1)
            y = y.transpose(1, 2).contiguous()

        return y.to(dtype=original_dtype)


class SSMBlock(nn.Module):
    def __init__(self, d_model, expansion, d_state, max_len, dropout=0.0):
        super().__init__()

        inner = d_model * expansion

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)

        self.ssm = DiagonalLTI_SSM(
            inner, d_state, max_len
        )

        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        z = self.norm(x)
        u, gate = self.in_proj(z).chunk(2, dim=-1)

        y = self.ssm(u)
        y = y * F.silu(gate)

        return residual + self.dropout(self.out_proj(y))


class SSLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        d_state,
        expansion,
        max_len
    ):
        super().__init__()

        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "d_state": d_state,
            "expansion": expansion,
            "max_len": max_len,
        }

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            SSMBlock(
                d_model,
                expansion,
                d_state,
                max_len
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Tied embeddings
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )

        return logits, loss


# --------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------

def load_model():
    with open(MODEL_DIR / "config.json", "r") as f:
        config = json.load(f)

    tokenizer = Tokenizer.from_file(
        str(MODEL_DIR / "tokenizer.json")
    )

    model = SSLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config["d_state"],
        expansion=config["expansion"],
        max_len=config["max_seq_len"],
    )

    state_dict = torch.load(
        MODEL_DIR / "model_state_dict.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, tokenizer


model, tokenizer = load_model()

EOS_ID = tokenizer.token_to_id("<|endoftext|>")
USER_ID = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")

MAX_SEQ_LEN = model.config["max_len"]


# --------------------------------------------------
# Text generation
# --------------------------------------------------

@torch.inference_mode()
def generate_answer(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40
):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    prompt_ids = (
        [USER_ID]
        + tokenizer.encode(prompt).ids
        + [EOS_ID, ASSISTANT_ID]
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        context = input_ids[:, -MAX_SEQ_LEN:]

        logits, _ = model(context)

        next_logits = logits[0, -1].float()

        next_logits = next_logits / max(temperature, 1e-5)

        if top_k is not None:
            values, _ = torch.topk(
                next_logits,
                min(top_k, next_logits.numel())
            )

            next_logits[next_logits < values[-1]] = -float("inf")

        probabilities = F.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        if next_token == EOS_ID:
            break

        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[next_token]], device=DEVICE)
            ],
            dim=1
        )

    generated_ids = input_ids[0, len(prompt_ids):].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()
