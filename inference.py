import torch
import torch.nn as nn
import cv2
import numpy as np
import base64

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

class SLRModel:
    def __init__(self, model_path='models/slr_model.pth'):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.classes = checkpoint['classes']
        self.model = Simple3DCNN(len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict(self, frames_b64):
        # Decode base64 frames and process
        frames = []
        for f in frames_b64[:16]:
            img_data = base64.b64decode(f)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (112, 112)) / 255.0
                frames.append(img)
        
        while len(frames) < 16:
            frames.append(frames[-1] if frames else np.zeros((112, 112, 3)))
        
        x = torch.FloatTensor(np.array(frames[:16])).permute(3, 0, 1, 2).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        
        return self.classes[pred.item()], conf.item()
