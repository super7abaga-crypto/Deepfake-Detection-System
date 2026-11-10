from fastapi import FastAPI

from backend.routes.image_routes import router as image_router


app = FastAPI(
    title="Deepfake Detection System",
    description="An AI-powered multimedia authenticity detection system.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to the Deepfake Detection System!",
        "status": "Application is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Deepfake Detection System"
    }


# Register image upload routes
app.include_router(image_router)