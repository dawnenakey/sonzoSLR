from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import boto3
import os
import uuid
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np
import base64
import cv2

app = FastAPI(title="SonZo.ai SLR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 14 * 14, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

MODEL_PATH = "models/slr_model.pth"
model = None
classes = []

try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    classes = checkpoint['classes']
    model = Simple3DCNN(len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: {len(classes)} classes")
except Exception as e:
    print(f"Model load error: {e}")

dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION', 'us-east-1'))
clients_table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_CLIENTS', 'sonzo-api-clients'))

class RecognizeRequest(BaseModel):
    frames: List[str]
    mode: str = "single"

class RecognizeResponse(BaseModel):
    sign: str
    confidence: float
    alternatives: Optional[List[dict]] = None
    processing_time_ms: float
    request_id: str

async def verify_api_key(x_api_key: str = Header(...)):
    try:
        response = clients_table.get_item(Key={"api_key": x_api_key})
        if "Item" not in response:
            raise HTTPException(status_code=401, detail="Invalid API key")
        client = response["Item"]
        if not client.get("active", False):
            raise HTTPException(status_code=403, detail="API key inactive")
        return client
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_frames(frames_b64):
    processed = []
    for f in frames_b64[:16]:
        try:
            if len(f) > 100:
                img_data = base64.b64decode(f)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (112, 112)) / 255.0
                    processed.append(img)
        except:
            pass
    if len(processed) == 0:
        processed = [np.random.rand(112, 112, 3) * 0.1 for _ in range(16)]
    while len(processed) < 16:
        processed.append(processed[-1] if processed else np.zeros((112, 112, 3)))
    x = torch.FloatTensor(np.array(processed[:16])).permute(3, 0, 1, 2).unsqueeze(0)
    return x

@app.get("/")
async def root():
    return {"service": "SonZo.ai SLR API", "version": "1.0.0", "model_loaded": model is not None, "num_classes": len(classes), "classes": classes, "patent": "Patent Pending 63/918,518"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "service": "SonZo.ai SLR API", "model_loaded": model is not None}

@app.post("/api/slr/recognize", response_model=RecognizeResponse)
async def recognize(request: RecognizeRequest, client: dict = Depends(verify_api_key)):
    import time
    start = time.time()
    request_id = str(uuid.uuid4())
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    x = process_frames(request.frames)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(1)
        top3 = torch.topk(probs, min(3, len(classes)), dim=1)
        alternatives = [{"sign": classes[idx], "confidence": round(score.item(), 4)} for score, idx in zip(top3.values[0], top3.indices[0])]
    sign = classes[pred.item()]
    confidence = round(conf.item(), 4)
    processing_time = round((time.time() - start) * 1000, 2)
    try:
        clients_table.update_item(Key={"api_key": client["api_key"]}, UpdateExpression="ADD calls_this_month :inc", ExpressionAttributeValues={":inc": 1})
    except:
        pass
    return RecognizeResponse(sign=sign.upper(), confidence=confidence, alternatives=alternatives, processing_time_ms=processing_time, request_id=request_id)

@app.get("/api/slr/usage")
async def get_usage(client: dict = Depends(verify_api_key)):
    tier_limits = {"basic": {"calls": 5000, "video_min": 500, "cost": 250}, "pro": {"calls": 25000, "video_min": 2500, "cost": 500}, "enterprise": {"calls": 999999, "video_min": 99999, "cost": 0}}
    tier = client.get("tier", "basic")
    limits = tier_limits.get(tier, tier_limits["basic"])
    return {"client_id": client["api_key"][:10] + "...", "client_name": client.get("client_name", "Unknown"), "tier": tier, "calls_this_month": int(client.get("calls_this_month", 0)), "calls_limit": limits["calls"], "cost_this_month": limits["cost"]}

@app.get("/api/slr/classes")
async def get_classes():
    return {"num_classes": len(classes), "classes": classes}
