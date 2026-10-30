# State Space Language Model (SSLM)

SSLM is a small, custom causal language model built from scratch with PyTorch. It
uses diagonal linear time-invariant (LTI) state-space layers instead of
self-attention and exposes the exported model through a FastAPI service and a
terminal-style web UI.

The repository contains:

- [`notebooks/training.ipynb`](/home/jashan/Projects/state_space_language_model/notebooks/training.ipynb):
  the Kaggle-oriented tokenizer, pretraining, supervised fine-tuning, evaluation,
  and export workflow;
- [`backend/chat.py`](/home/jashan/Projects/state_space_language_model/backend/chat.py):
  the inference-time model definition and top-k temperature sampling;
- [`backend/main.py`](/home/jashan/Projects/state_space_language_model/backend/main.py):
  the FastAPI application;
- [`frontend/`](/home/jashan/Projects/state_space_language_model/frontend):
  the browser terminal interface; and
- [`backend/sslm/`](/home/jashan/Projects/state_space_language_model/backend/sslm):
  the exported weights, tokenizer, and architecture configuration used by the
  service.

## Run the application with Docker

The easiest way to run the checked-in model is Docker Compose. Docker and the
Compose plugin are required.

```bash
git clone https://github.com/jashan7305/state_space_language_model.git
cd state_space_language_model
docker compose up --build
```

Once the image has started, open <http://127.0.0.1:8007>. The Compose file maps
host port `8007` to the container's port `8000` and binds it to localhost.
Inference runs on the CPU; the compose configuration limits OpenMP and MKL to
four threads.

To stop the service:

```bash
docker compose down
```

Useful checks:

```bash
curl http://127.0.0.1:8007/health
curl -X POST http://127.0.0.1:8007/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what a state space model is.","max_new_tokens":80}'
```

The API also provides `GET /answer`, which returns the most recently generated
prompt and answer. Only one generation is allowed at a time by the backend
generation lock.

## Local backend development

The backend targets Python 3.11 or newer and uses `uv` for dependency
management. From the repository root:

```bash
cd backend
uv sync
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The model is loaded when
[`chat.py`](/home/jashan/Projects/state_space_language_model/backend/chat.py)
is imported, so startup requires the three files in `backend/sslm/`.

## Model architecture

The checked-in configuration is a roughly 100M-parameter model:

| Component | Value |
| --- | ---: |
| Vocabulary | 32,000 BPE tokens |
| Model width (`d_model`) | 768 |
| SSM blocks | 14 |
| State dimension (`d_state`) | 16 |
| Inner expansion | 3x (2,304 channels) |
| Maximum context | 512 tokens |
| Parameters | 100,485,888 |
| Device in the service | CPU |

Each `SSMBlock` performs:

1. RMSNorm on the residual stream.
2. A bias-free projection from 768 channels to `2 * (768 * 3)` channels.
3. A split into an SSM input (`u`) and a SiLU gate.
4. A diagonal LTI SSM over the expanded channels.
5. Elementwise gated activation, output projection back to 768 channels, and a
   residual connection.

The final RMSNorm is followed by a vocabulary projection. Input embeddings and
the language-model head share the same weight matrix.

### How the state-space layer works

For each expanded channel and state, the layer learns:

- positive decay rates, represented as `softplus(log_rates) + 1e-4`;
- input coefficients `B`;
- output coefficients `C`; and
- a direct/skip coefficient `D`.

For a sequence, it constructs the time-domain kernel

```text
k[c, t] = sum_n B[c, n] * C[c, n] * exp(-rate[c, n] * t)
```

and applies the resulting causal convolution with an FFT. The direct path
`D * u` is then added. This gives the model a long-range sequence-mixing
operation without attention's quadratic pairwise token interactions.

This is a **diagonal, non-selective LTI SSM**, not selective Mamba: the
dynamics and convolution kernel do not change as a function of the current
token. The implementation uses float32 for the FFT computation and restores
the original input dtype afterward.

The model is trained as a causal next-token predictor. Given token IDs with
shape `(batch, sequence)`, it returns logits with shape
`(batch, sequence, 32000)`. The loss compares each position with the next
token; labels equal to `-100` are ignored during supervised fine-tuning.

## Tokenization, training, and fine-tuning

The notebook trains a 32K Byte Pair Encoding tokenizer from normalized,
deduplicated text and reserves these special tokens:

```text
<|pad|> <|unk|> <|endoftext|> <|user|> <|assistant|> <|system|>
```

The notebook's intended workflow is:

1. Enable Internet and a GPU in Kaggle.
2. Stream FineWeb and FineWeb-Edu rather than loading the full datasets into
   memory.
3. Train the tokenizer and write tokenized `uint16` memory-mapped streams.
4. Pretrain with AdamW, gradient accumulation, warmup, cosine decay, gradient
   clipping, mixed precision, validation, and resumable checkpoints.
5. Convert UltraChat conversations and English OASST1 parent/child replies into
   role-tagged examples.
6. Mask user and system tokens with `-100`, so supervised loss is computed only
   on assistant content and its end-of-text marker.
7. Export `model_state_dict.pt`, `tokenizer.json`, and `config.json` for
   inference.

The active values in the notebook's configuration cell are:

- 15M FineWeb training tokens;
- 10M FineWeb-Edu training tokens;
- 1M validation tokens;
- micro-batch size 2, gradient accumulation 8, and sequence length 512
  (8,192 tokens per optimizer update);
- 2M target assistant tokens for SFT; and
- one SFT epoch.

The opening notebook markdown describes a larger 500M-token pretraining run and
an 8M-token SFT budget, but those are not the current executable values. Change
the configuration cell before claiming or running those larger budgets. The
full training workflow is GPU- and storage-intensive and is separate from the
Docker inference deployment.

## Inference format

For a prompt, the service creates this token sequence:

```text
<|user|> prompt <|endoftext|> <|assistant|>
```

It repeatedly evaluates the last context, samples from the logits using
temperature and top-k filtering, and stops at `<|endoftext|>` or
`max_new_tokens`. If the growing sequence exceeds 512 tokens, inference keeps
the most recent 512 tokens. The current implementation recomputes the full
context for every generated token; it is simple and correct for this export,
but it does not use a recurrent state cache or streaming responses.

The `/prompt` request accepts:

```json
{
  "prompt": "Hello",
  "max_new_tokens": 150,
  "temperature": 0.8,
  "top_k": 40
}
```

`max_new_tokens` is limited to 256 by the API schema, temperature must be in
`(0, 2]`, and prompts must be non-empty.

## Repository layout

```text
.
├── backend/
│   ├── chat.py                 # Model, loading, and generation
│   ├── main.py                 # FastAPI routes and static-file serving
│   ├── Dockerfile              # CPU inference image
│   ├── pyproject.toml          # Python dependencies
│   └── sslm/                   # Exported model artifacts
├── compose.yaml                # Docker Compose service
├── frontend/                   # Terminal UI
└── notebooks/training.ipynb    # Kaggle training/export notebook
```

## Limitations

- The model is intentionally small and trained on a limited budget; fluent,
  factual, or safe responses are not guaranteed.
- It has a 512-token maximum context and CPU inference is relatively slow.
- The service exposes a single in-memory latest answer and has no authentication,
  persistence, batching, or streaming.
- The notebook is designed for Kaggle GPU sessions. Checkpoints and datasets
  under `/kaggle/working` must be saved or published explicitly before a
  session ends.