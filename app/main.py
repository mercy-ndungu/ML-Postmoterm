from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analysis, upload

app = FastAPI(
    title="ML Postmortem API",
    description="AI-powered ML experiment reviewer — upload a dataset, get a plain-English postmortem.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])


@app.get("/")
def root():
    return {"message": "ML Postmortem API is running. POST to /api/analyze to get started."}
