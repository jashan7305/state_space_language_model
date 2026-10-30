
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "dumbasslm"

DEVICE = torch.device("cpu")

# Limit CPU threads if needed.
torch.set_num_threads(4)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Initialize once when imported
# --------------------------------------------------

print("[DumbassLM] Loading DumbassLM tokenizer...")
tokenizer = load_tokenizer()

print("[DumbassLM] Loading DumbassLM model...")
model = load_model()

print("[DumbassLM] Model loaded successfully.")

EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id

if PAD_ID is None:
    PAD_ID = EOS_ID


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

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Tokenize using SmolLM2's chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": PAD_ID,
        "eos_token_id": EOS_ID,
    }

    if temperature > 0:
        generation_kwargs["temperature"] = temperature

        if top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = top_k

    outputs = model.generate(
        **inputs,
        **generation_kwargs
    )

    # Extract only newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

