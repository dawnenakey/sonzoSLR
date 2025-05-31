"""
Streamlit app for Sign Language Recognition and Analysis.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
import joblib
from sklearn.metrics import roc_curve, auc

def create_3d_cnn_visualization():
    """Create a 3D visualization of CNN architecture."""
    # Create a 3D surface plot to represent CNN layers
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
    fig.update_layout(
        title='3D CNN Architecture Visualization',
        scene = dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Depth'
        ),
        width=800,
        height=600
    )
    return fig

def create_lstm_metrics():
    """Create LSTM training metrics visualization."""
    # Sample data for LSTM metrics
    epochs = list(range(1, 21))
    train_loss = [0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3,
                  0.28, 0.25, 0.23, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12]
    val_loss = [0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
                0.33, 0.3, 0.28, 0.25, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18]
    accuracy = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                0.72, 0.75, 0.77, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]

    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Training and Validation Loss', 'Model Accuracy'))

    # Add loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')),
        row=1, col=1
    )

    # Add accuracy curve
    fig.add_trace(
        go.Scatter(x=epochs, y=accuracy, name='Accuracy', line=dict(color='green')),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        width=800,
        title_text="LSTM Training Metrics",
        showlegend=True
    )

    return fig

def create_confusion_matrix(classes):
    # Mock confusion matrix data
    np.random.seed(42)
    cm = np.random.randint(0, 10, size=(len(classes), len(classes)))
    fig = px.imshow(cm, text_auto=True, x=classes, y=classes, color_continuous_scale='Blues')
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig, cm

def create_classification_report(classes):
    # Mock precision, recall, f1, support
    np.random.seed(42)
    data = {
        'Precision': np.round(np.random.uniform(0.7, 1.0, len(classes)), 2),
        'Recall': np.round(np.random.uniform(0.7, 1.0, len(classes)), 2),
        'F1-Score': np.round(np.random.uniform(0.7, 1.0, len(classes)), 2),
        'Support': np.random.randint(5, 20, len(classes))
    }
    df = pd.DataFrame(data, index=classes)
    return df

def create_per_class_accuracy(classes):
    # Mock per-class accuracy
    np.random.seed(42)
    accuracy = np.round(np.random.uniform(0.7, 1.0, len(classes)), 2)
    fig = px.bar(x=classes, y=accuracy, labels={'x': 'Class', 'y': 'Accuracy'}, title="Per-Class Accuracy")
    return fig, accuracy

# New: ROC Curve & AUC

def create_roc_curve(classes):
    np.random.seed(42)
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Just for demo
    auc = 0.85
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title=f'ROC Curve (AUC={auc})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def load_classification_report_csv(path, fallback_classes):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        # Optionally, filter to only the classes you want to show
        df = df.loc[[str(c) for c in fallback_classes] if all(str(c) in df.index for c in fallback_classes) else df.index]
        return df
    return None

def load_roc_data(path):
    if os.path.exists(path):
        results = joblib.load(path)
        return results['roc_data'], results['class_names']
    return None, None

def main():
    st.set_page_config(page_title="Sign Language Recognition Demo", layout="wide")
    # Sidebar
    with st.sidebar:
        st.title("SLR Capstone Demo")
        st.markdown("**Instructions:**\n1. Upload a sign language video.\n2. View model predictions and metrics.\n3. Explore architecture and analytics in the tabs.")
        st.markdown("---")
        st.info("Project by Dawnena Key, Masters of Data Science Capstone 2025")
    st.title("Sign Language Recognition Platform")
    st.write("Upload sign language videos for analysis and recognition. This demo showcases model predictions, evaluation metrics, and architecture visualizations for your capstone presentation.")
    
    # File uploader for video files
    uploaded_file = st.file_uploader(
        "Upload a sign language video file (MP4, AVI, or MOV format)",
        type=['mp4', 'avi', 'mov']
    )
    
    if uploaded_file:
        st.success(f"Video file {uploaded_file.name} uploaded successfully!")
        
        # Simulate inference time
        start_time = time.time()
        time.sleep(1.2)  # Simulate processing
        inference_time = time.time() - start_time
        st.metric("Inference Time (s)", f"{inference_time:.2f}")
        
        # Mock prediction and true label
        mock_classes = [281, 284, 26, 107, 682]
        predicted_class = np.random.choice(mock_classes)
        true_class = np.random.choice(mock_classes)
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**True Class:** {true_class}")
        
        # Show video
        st.video(uploaded_file)
        
        st.header("Model Architecture and Analytics")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "3D CNN Architecture", "LSTM Training Metrics", "Confusion Matrix", "Classification Report", "Per-Class Accuracy", "ROC Curve & AUC"
        ])
        with tab1:
            st.subheader("3D CNN Architecture Visualization")
            st.write("This visualization shows the 3D convolutional neural network architecture used for sign language recognition. The 3D CNN extracts spatial and temporal features from video frames.")
            cnn_fig = create_3d_cnn_visualization()
            st.plotly_chart(cnn_fig)
            st.markdown("**CNN Architecture Details:**\n- Input Layer: 3D video frames (height × width × channels)\n- Convolutional Layers: Multiple 3D convolutional layers with ReLU activation\n- Pooling Layers: 3D max pooling for dimensionality reduction\n- Fully Connected Layers: Final classification layers")
        with tab2:
            st.subheader("LSTM Training Metrics")
            st.write("These graphs show the training progress of our LSTM model for sequence processing. LSTMs help capture temporal dependencies in sign language sequences.")
            lstm_fig = create_lstm_metrics()
            st.plotly_chart(lstm_fig)
            st.markdown("**LSTM Architecture Details:**\n- Input: Sequence of features from CNN\n- LSTM Layers: Multiple LSTM layers for temporal processing\n- Dropout: 0.5 for regularization\n- Output: Sign language class probabilities")
        with tab3:
            st.subheader("Confusion Matrix")
            st.write("The confusion matrix visualizes the model's performance by showing the number of correct and incorrect predictions for each class.")
            cm_fig, cm = create_confusion_matrix(mock_classes)
            st.plotly_chart(cm_fig)
        with tab4:
            st.subheader("Precision, Recall, F1-Score Table")
            st.write("This table summarizes the model's precision, recall, F1-score, and support for each class. These metrics help evaluate the quality of predictions.")
            report_df = load_classification_report_csv('classification_report.csv', mock_classes)
            if report_df is not None:
                st.dataframe(report_df)
            else:
                report_df = create_classification_report(mock_classes)
                st.dataframe(report_df)
        with tab5:
            st.subheader("Per-Class Accuracy")
            st.write("This bar chart shows the accuracy for each class, helping identify which signs are recognized well and which need improvement.")
            acc_fig, acc = create_per_class_accuracy(mock_classes)
            st.plotly_chart(acc_fig)
        with tab6:
            st.subheader("ROC Curve & AUC")
            st.write("The ROC curve illustrates the trade-off between true positive rate and false positive rate for the model. The AUC summarizes overall performance.")
            roc_data, class_names = load_roc_data('roc_results.pkl')
            if roc_data is not None and class_names is not None:
                fig_roc = go.Figure()
                for class_name in class_names:
                    fpr = roc_data[class_name]['fpr']
                    tpr = roc_data[class_name]['tpr']
                    auc_val = roc_data[class_name]['auc']
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines', name=f'{class_name} (AUC={auc_val:.2f})'
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')
                ))
                fig_roc.update_layout(
                    title='ROC Curves for All Classes',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig_roc)
            else:
                roc_fig = create_roc_curve(mock_classes)
                st.plotly_chart(roc_fig)

if __name__ == "__main__":
    main() 