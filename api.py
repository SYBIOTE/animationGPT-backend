"""
AnimationGPT / MotionGPT microservice: JSON-only API for text-to-motion.
Same contract as HY-Motion: POST /v1/motion and GET /health for Next.js / SaaS clients.

Usage (from this repo root, with venv + deps installed):
    python -m uvicorn api:app --host 0.0.0.0 --port 8082

Env:
    MOTION_FPS: Override output fps in JSON (default: 20, HumanML3D / motion_json default)
    T2M_CACHE_FOLDER: Optional folder passed to T2MBot for checkpoints output (default: cache)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SERVER = _ROOT / "server"
for p in (_SERVER, _ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bot import T2MBot
from motion_json import DEFAULT_FPS, joints_to_motion_data

MOTION_FPS = float(os.environ.get("MOTION_FPS", str(DEFAULT_FPS)))
_CACHE_FOLDER = os.environ.get("T2M_CACHE_FOLDER", "cache")

_bot: T2MBot | None = None


def get_bot() -> T2MBot:
    global _bot
    if _bot is None:
        # cwd-relative paths in MotionGPT configs expect running from repo root or server
        os.chdir(_ROOT)
        _bot = T2MBot(folder=_CACHE_FOLDER)
    return _bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_bot()
    except Exception as e:
        print(f">>> [WARNING] T2MBot not loaded at startup: {e}")
    yield


app = FastAPI(title="AnimationGPT API", version="1.0", lifespan=lifespan)


class MotionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    duration: float = Field(default=3.0, ge=0.5, le=30.0)
    seed: int = Field(default=42, ge=0)
    cfg_scale: float = Field(default=5.0, ge=1.0, le=20.0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/motion")
def generate_motion(req: MotionRequest):
    try:
        bot = get_bot()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

    text = req.text.strip()
    try:
        joints, lengths, _feats = bot.infer_motion_tensors(text)
        if lengths <= 0:
            raise HTTPException(status_code=503, detail="Model returned empty motion.")
        j = joints[:lengths].detach().cpu().numpy()
        motion = joints_to_motion_data(j, lengths, fps=MOTION_FPS)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation failed: {e}")

    meta = {
        "text": text,
        "duration": req.duration,
        "seed": req.seed,
        "provider": "animationgpt",
    }
    return {"motion": motion, "meta": meta}
