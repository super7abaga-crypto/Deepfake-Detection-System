
from fastapi import FastAPI

# Create our application
app = FastAPI(
    title="Deepfake Detection System",
    description="An AI-powered multimedia authenticity detection system.",
    version="1.0.0"
)


# Homepage endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the Deepfake Detection System!",
        "status": "Application is running"
    }


# Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Deepfake Detection System"
    }
