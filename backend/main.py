
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat import generate_answer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SSLM API",
    description="Custom State Space Language Model",
    version="1.0.0"
)


# --------------------------------------------------
# State
# --------------------------------------------------

latest_result = None
generation_lock = Lock()


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
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
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "SSLM",
        "device": "cpu"
    }


@app.post("/prompt")
def submit_prompt(request: PromptRequest):
    global latest_result

    try:
        with generation_lock:
            answer = generate_answer(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k
            )

        latest_result = {
            "prompt": request.prompt,
            "answer": answer
        }

        return {
            "status": "success",
            **latest_result
        }

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
