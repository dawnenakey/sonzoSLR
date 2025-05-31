import torch
import numpy as np
import os
from src.train import SignLanguageModel, SignLanguageDataset
from torch.utils.data import DataLoader

# Paths
MODEL_PATH = 'best_model.pth'
DATA_DIR = 'data/raw'
OUTPUT_DIR = 'reports'
BATCH_SIZE = 32
NUM_CLASSES = 1000  # Adjust if needed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load validation set
    dataset = SignLanguageDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load model
    model = SignLanguageModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, 'y_true.npy'), np.array(y_true))
    np.save(os.path.join(OUTPUT_DIR, 'y_pred.npy'), np.array(y_pred))

    # Save class names if available (placeholder: 0, 1, 2, ...)
    class_names = [str(i) for i in range(NUM_CLASSES)]
    np.save(os.path.join(OUTPUT_DIR, 'class_names.npy'), np.array(class_names))
    print('Saved y_true, y_pred, and class_names to', OUTPUT_DIR)

if __name__ == '__main__':
    main() 