import numpy as np
from sklearn.metrics import roc_curve, auc
import joblib


np.random.seed(42)
n_samples = 100
class_names = [281, 284, 26, 107, 682]
n_classes = len(class_names)
y_true = np.random.choice(class_names, size=n_samples)
y_proba = np.random.rand(n_samples, n_classes)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize to sum to 1


roc_data = {}
for i, class_name in enumerate(class_names):
    y_true_bin = (y_true == class_name).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data[class_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

joblib.dump({'roc_data': roc_data, 'class_names': class_names}, 'roc_results.pkl')
print("roc_results.pkl has been created!")
