
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sslm_chat import generate_answer as generate_sslm
from dumbasslm_chat import generate_answer as generate_dumbasslm


# --------------------------------------------------
# Paths
# --------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="SSLM Terminal",
    description="Compare a custom SSM against a small Transformer.",
    version="1.0.0"
)


# --------------------------------------------------
# Model registry
# --------------------------------------------------

MODELS = {
    "sslm": {
        "name": "SSLM-100M",
        "generate": generate_sslm
    },
    "dumbasslm": {
        "name": "DumbassLM-135M",
        "generate": generate_dumbasslm
    }
}

generation_lock = Lock()
latest_result = None


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: Literal["sslm", "dumbasslm"] = "sslm"

    max_new_tokens: int = Field(default=150, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_k: int = Field(default=40, ge=1)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": [
            {
                "id": model_id,
                "name": model["name"]
            }
            for model_id, model in MODELS.items()
        ],
        "device": "cpu"
    }


@app.post("/prompt")
def submit_prompt(request: PromptRequest):
    global latest_result

    selected_model = MODELS[request.model]

    try:
        with generation_lock:
            answer = selected_model["generate"](
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )

        latest_result = {
            "status": "success",
            "model": request.model,
            "model_name": selected_model["name"],
            "prompt": request.prompt,
            "answer": answer
        }

        return latest_result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/answer")
def get_answer():
    if latest_result is None:
        raise HTTPException(
            status_code=404,
            detail="No answer generated yet."
        )

    return latest_result
