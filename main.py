"""
SonZo.ai - Sign Language Recognition API
Patent Pending 63/918,518 | Indigenous-owned technology company

This is the main FastAPI application for the SonZo SLR Licensing API.
"""
import os
import time
import uuid
import logging
import random
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE_CLIENTS = os.getenv("DYNAMODB_TABLE_CLIENTS", "sonzo-api-clients")
DYNAMODB_TABLE_USAGE = os.getenv("DYNAMODB_TABLE_USAGE", "sonzo-api-usage")

# Pricing Tiers - SonZo.ai
PRICING_TIERS = {
    "basic": {
        "limit": 5000,
        "video_minutes": 500,
        "streaming_minutes": 100,
        "price": 250
    },
    "pro": {
        "limit": 25000,
        "video_minutes": 2500,
        "streaming_minutes": 1000,
        "price": 500
    },
    "enterprise": {
        "limit": float("inf"),
        "video_minutes": float("inf"),
        "streaming_minutes": float("inf"),
        "price": 0  # Custom pricing
    }
}

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
clients_table = dynamodb.Table(DYNAMODB_TABLE_CLIENTS)
usage_table = dynamodb.Table(DYNAMODB_TABLE_USAGE)

# =============================================================================
# Request/Response Models
# =============================================================================

class RecognizeRequest(BaseModel):
    frames: List[str] = Field(..., description="Base64 encoded video frames")
    return_alternatives: bool = Field(default=False, description="Return alternative interpretations")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum confidence threshold")

class RecognizeResponse(BaseModel):
    sign: str
    confidence: float
    alternatives: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float
    request_id: str

class UsageResponse(BaseModel):
    client_id: str
    client_name: str
    tier: str
    calls_this_month: int
    calls_limit: int
    video_minutes_used: int
    video_minutes_limit: int
    streaming_minutes_used: int
    streaming_minutes_limit: int
    cost_this_month: float

class SignsResponse(BaseModel):
    signs: List[str]
    count: int

# Supported ASL signs (placeholder - your real model will have more)
SUPPORTED_SIGNS = [
    "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "LOVE",
    "FRIEND", "FAMILY", "WORK", "HOME", "EAT", "DRINK", "GOOD", "BAD",
    "WANT", "NEED", "LIKE", "NAME", "WHAT", "WHERE", "WHEN", "WHY", "HOW",
    "YOU", "ME", "THEY", "WE", "HE", "SHE", "THIS", "THAT", "HERE", "THERE"
]

# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> Dict:
    """Verify API key and return client data."""
    try:
        response = clients_table.get_item(Key={"api_key": x_api_key})
        if "Item" not in response:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        client = response["Item"]
        if not client.get("active", True):
            raise HTTPException(status_code=403, detail="API key deactivated")
        
        # Check usage limits
        tier = client.get("tier", "basic")
        calls = int(client.get("calls_this_month", 0))
        limit = PRICING_TIERS.get(tier, PRICING_TIERS["basic"])["limit"]
        
        if calls >= limit and limit != float("inf"):
            raise HTTPException(
                status_code=429,
                detail=f"Monthly limit exceeded. Upgrade your plan at https://sonzo.ai/pricing"
            )
        
        return client
    except ClientError as e:
        logger.error(f"DynamoDB error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")

# =============================================================================
# Usage Tracking
# =============================================================================

async def log_usage(client_id: str, endpoint: str, request_id: str, video_seconds: int = 0):
    """Log API usage to DynamoDB."""
    try:
        now = datetime.now(timezone.utc)
        
        # Log individual request
        usage_table.put_item(Item={
            "client_id": client_id,
            "timestamp": now.isoformat(),
            "month": now.strftime("%Y-%m"),
            "endpoint": endpoint,
            "request_id": request_id,
            "video_seconds": video_seconds
        })
        
        # Update client counters
        update_expr = "SET calls_this_month = if_not_exists(calls_this_month, :zero) + :inc, last_call = :now"
        expr_values = {":inc": 1, ":zero": 0, ":now": now.isoformat()}
        
        if video_seconds > 0:
            update_expr += ", video_seconds_this_month = if_not_exists(video_seconds_this_month, :zero) + :video"
            expr_values[":video"] = video_seconds
        
        clients_table.update_item(
            Key={"api_key": client_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values
        )
    except ClientError as e:
        logger.error(f"Usage logging failed: {e}")

# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("🚀 SonZo.ai SLR API starting...")
    logger.info(f"   Region: {AWS_REGION}")
    logger.info(f"   Clients table: {DYNAMODB_TABLE_CLIENTS}")
    logger.info(f"   Usage table: {DYNAMODB_TABLE_USAGE}")
    yield
    logger.info("👋 SonZo.ai SLR API stopped")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="SonZo.ai SLR API",
    description="Sign Language Recognition API - Patent Pending 63/918,518",
    version="1.0.0",
    contact={
        "name": "SonZo API Support",
        "url": "https://sonzo.ai",
        "email": "api@sonzo.ai"
    },
    license_info={
        "name": "Commercial License",
        "url": "https://sonzo.ai/license"
    },
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "SonZo.ai SLR API",
        "version": "1.0.0",
        "status": "healthy",
        "documentation": "https://sonzo.ai/docs",
        "patent": "Patent Pending 63/918,518"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "SonZo.ai SLR API"
    }

@app.post("/api/slr/recognize", response_model=RecognizeResponse)
async def recognize(
    request: RecognizeRequest,
    background_tasks: BackgroundTasks,
    client: Dict = Depends(verify_api_key)
):
    """
    Recognize ASL signs from video frames.
    
    Send base64-encoded video frames and receive the recognized sign.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Validate input
    if len(request.frames) < 1 or len(request.frames) > 30:
        raise HTTPException(status_code=400, detail="1-30 frames required")
    
    # TODO: Replace with actual SLR inference
    # This is placeholder logic - integrate your 3D CNN + LSTM model here
    main_sign = random.choice(SUPPORTED_SIGNS)
    confidence = random.uniform(0.75, 0.98)
    
    alternatives = []
    if request.return_alternatives:
        alternatives = [
            {"sign": s, "confidence": round(random.uniform(0.1, 0.5), 4)}
            for s in random.sample(SUPPORTED_SIGNS, min(3, len(SUPPORTED_SIGNS)))
            if s != main_sign
        ]
        alternatives = sorted(alternatives, key=lambda x: x["confidence"], reverse=True)
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Log usage in background
    background_tasks.add_task(
        log_usage,
        client["api_key"],
        "/api/slr/recognize",
        request_id
    )
    
    # Build response
    response = RecognizeResponse(
        sign=main_sign if confidence >= request.confidence_threshold else "UNKNOWN",
        confidence=round(confidence, 4),
        processing_time_ms=round(processing_time_ms, 2),
        request_id=request_id,
        alternatives=alternatives if request.return_alternatives else None
    )
    
    return response

@app.get("/api/slr/usage", response_model=UsageResponse)
async def get_usage(client: Dict = Depends(verify_api_key)):
    """Get current usage statistics for your API key."""
    tier = client.get("tier", "basic")
    tier_info = PRICING_TIERS.get(tier, PRICING_TIERS["basic"])
    
    calls = int(client.get("calls_this_month", 0))
    video_seconds = int(client.get("video_seconds_this_month", 0))
    streaming_seconds = int(client.get("streaming_seconds_this_month", 0))
    
    return UsageResponse(
        client_id=client["api_key"][:8] + "...",
        client_name=client.get("client_name", "Unknown"),
        tier=tier,
        calls_this_month=calls,
        calls_limit=int(tier_info["limit"]) if tier_info["limit"] != float("inf") else -1,
        video_minutes_used=video_seconds // 60,
        video_minutes_limit=int(tier_info["video_minutes"]) if tier_info["video_minutes"] != float("inf") else -1,
        streaming_minutes_used=streaming_seconds // 60,
        streaming_minutes_limit=int(tier_info["streaming_minutes"]) if tier_info["streaming_minutes"] != float("inf") else -1,
        cost_this_month=float(tier_info["price"])
    )

@app.get("/api/slr/signs", response_model=SignsResponse)
async def list_signs(client: Dict = Depends(verify_api_key)):
    """Get list of all supported ASL signs."""
    return SignsResponse(
        signs=SUPPORTED_SIGNS,
        count=len(SUPPORTED_SIGNS)
    )

# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail
            }
        }
    )

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
