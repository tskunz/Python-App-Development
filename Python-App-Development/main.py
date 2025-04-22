from fastapi import FastAPI, Depends, HTTPException
from typing import List, Dict
import mlflow
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ML Applications API",
    description="API for ML applications including recommendations and batch predictions",
    version="1.0.0"
)

# Import routes
from routes import recommendations, batch_predictions, events

# Register routes
app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(batch_predictions.router, prefix="/batch", tags=["batch"])
app.include_router(events.router, prefix="/events", tags=["events"])

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "ML Applications API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 