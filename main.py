from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(
    title="Image Question API",
    description="API that reads questions from images using OpenAI model via Agno",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

app.include_router(api_router, prefix="/api", tags=["Agent"])


@app.get("/")
def root():
    return {"message": "Image Question API is running!"}
