import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_performance_summary(train_losses, val_losses, val_accuracies):
    """Create a combined plot of training progress."""
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:10],  # Show first 10 classes
                yticklabels=class_names[:10])
    plt.title('Confusion Matrix (Top 10 Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_handshape_distribution(df, handshape_col):
    """Create handshape distribution visualization."""
    plt.figure(figsize=(12, 6))
    df[handshape_col].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Most Common Handshapes')
    plt.xlabel('Handshape')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/handshape_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_by_category(df, accuracy_by_category):
    """Create performance breakdown by category."""
    plt.figure(figsize=(10, 6))
    categories = list(accuracy_by_category.keys())
    accuracies = list(accuracy_by_category.values())
    
    plt.bar(categories, accuracies)
    plt.title('Model Performance by Sign Category')
    plt.xlabel('Category')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/performance_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Load your data and model results here
    # This is a placeholder - replace with your actual data loading code
    train_losses = [0.8, 0.6, 0.5, 0.4, 0.3]  # Example data
    val_losses = [0.85, 0.65, 0.55, 0.45, 0.35]
    val_accuracies = [70, 75, 80, 85, 87]
    
    # Create visualizations
    create_performance_summary(train_losses, val_losses, val_accuracies)
    
    # Example data for other visualizations
    df = pd.read_csv('src/data/asl_lex/signdata.csv', encoding='latin1')
    print('Columns in signdata.csv:', df.columns.tolist())
    create_handshape_distribution(df, 'Handshape.2.0')
    
    # Example performance by category
    accuracy_by_category = {
        'Simple Signs': 92,
        'Complex Signs': 85,
        'Two-Handed': 88,
        'One-Handed': 90
    }
    create_performance_by_category(df, accuracy_by_category)

if __name__ == "__main__":
    main() 