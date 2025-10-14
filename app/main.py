import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.routes.qa import router as qa_router
from app.llm import get_llm
from app.logging import setup_logging
from app.middlewares import RequestIDMiddleware, AccessLogMiddleware

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)
get_llm()

app = FastAPI(title="Fabrication ChatLLM", version="1.0")

LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

FRONTEND_URL = os.getenv("FRONTEND_URL")

ALLOWED_ORIGINS = [
    o for o in LOCAL_ORIGINS + ([FRONTEND_URL] if FRONTEND_URL else []) if o
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https:\/\/[a-zA-Z0-9\-\._]+\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Trace-ID"],
)

TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "*").split(",")
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=[h.strip() for h in TRUSTED_HOSTS if h.strip()]
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(AccessLogMiddleware)

app.include_router(qa_router)
