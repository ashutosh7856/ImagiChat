from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(
    title="Image Question API",
    description="API that reads questions from images using OpenAI model via Agno",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api", tags=["Agent"])


@app.get("/")
def root():
    return {"message": "Image Question API is running!"}
