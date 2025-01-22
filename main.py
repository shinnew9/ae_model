import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from data_processor import load_data, normalize_data

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from autoencoder import Autoencoder
from mahala import calc_inv_cov, loss_function_mahala

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "label_done_data")
LIGHT_DIR = os.path.join(DATA_DIR, "light")
HEAVY_DIR = os.path.join(DATA_DIR, "heavy")
CONTEXT_FRAMES = 5  # Number of consecutive frames used
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
WINDOW_SIZE = 3 * SAMPLE_RATE  # 3-second window
STEP_SIZE = WINDOW_SIZE // 2   # 50% overlap


if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def split_data_into_windows(data, context_frames):
    '''
    Splits data into context frame windows
    '''
    windowed_data = []
    step_size = context_frames // 2
    total_frames = data.shape[0]

    for start_idx in range(0, total_frames - context_frames + 1, step_size):
        window = data[start_idx:start_idx + context_frames, :]
        windowed_data.append(window.flatten())  # Flatten 2D array to 1D
    return np.array(windowed_data)


def train(model, train_loader, optimizer):
    '''
    autoencoder 모델을 학습
    '''
    model.train()
    train_loss = 0
    for batch in train_loader:
        data = batch[0].to(DEVICE)  # Input data shape: [batch_size, input_dim]
        optimizer.zero_grad()
        recon, _ = model(data)  # Autoencoder forward pass

        # Ensure shapes match
        if recon.shape != data.shape:
            recon = recon.view_as(data)

        loss = F.mse_loss(recon, data)  # MSE Loss between input and reconstruction
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Debugging print
        # print(f"Input shape: {data.shape}, Recon shape: {recon.shape}")  
    return train_loss / len(train_loader)


def evaluate_component(model, normal_path, abnormal_path, threshold, block_size=8):
    '''
    Evaluates the trained autoencoder on a component
    '''
    model.eval()
    normal_data, _ = load_data(normal_path, abnormal_path, CONTEXT_FRAMES, normalize_method="minmax")
    abnormal_data, _ = load_data(abnormal_path, abnormal_path, CONTEXT_FRAMES, normalize_method="minmax")

    normal_data = split_data_into_windows(normal_data, CONTEXT_FRAMES)
    abnormal_data = split_data_into_windows(abnormal_data, CONTEXT_FRAMES)

    normal_loader = DataLoader(TensorDataset(torch.tensor(normal_data, dtype=torch.float32)), batch_size=BATCH_SIZE)
    abnormal_loader = DataLoader(TensorDataset(torch.tensor(abnormal_data, dtype=torch.float32)), batch_size=BATCH_SIZE)

    anomaly_scores = []
    decisions = []
    y_true = []

    start_time = time.time()
    with torch.no_grad():
        for loader, label in [(normal_loader, 0), (abnormal_loader, 1)]:
            for batch in loader:
                data = batch[0].to(DEVICE)
                recon, _ = model(data)

                # Ensure shape consistency
                if recon.shape != data.shape:
                    recon = recon.view_as(data)

                loss = F.mse_loss(recon, data, reduction='none').mean(dim=1)
                scores = loss.cpu().numpy()
                anomaly_scores.extend(scores)
                decisions.extend([1 if score > threshold else 0 for score in scores])
                y_true.extend([label] * len(scores))
    total_time = time.time() - start_time
    return anomaly_scores, decisions, y_true, total_time


def save_results(component, scores, decisions, y_true):
    '''
    Saves evaluation results and prints metrics
    '''
    auc = roc_auc_score(y_true, scores)
    precision = precision_score(y_true, decisions, zero_division=0)
    recall = recall_score(y_true, decisions, zero_division=0)
    f1 = f1_score(y_true, decisions, zero_division=0)

    print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    result_file = os.path.join(RESULTS_DIR, f"{component}_results.csv")

    np.savetxt(result_file, np.c_[scores, decisions, y_true], delimiter=",",
               header="Score,Decision,TrueLabel", comments="")
    print(f"Results saved to {result_file}")

    return auc, precision, recall, f1


def main():
    print("Starting main...")
    input_dim = CONTEXT_FRAMES * 128  # Flattened input dimension
    model = Autoencoder(input_dim=input_dim, block_size=8).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Model Summary
    print("Model Summary:")
    summary(model, input_size=(1, input_dim))

    # Training
    print("Loading training data...")
    normal_data, _ = load_data(LIGHT_DIR, LIGHT_DIR, CONTEXT_FRAMES, normalize_method="minmax")
    normal_data = split_data_into_windows(normal_data, CONTEXT_FRAMES)

    train_data = torch.tensor(normal_data, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

    print("Training Autoencoder...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer)
        print(f"Epoch [{epoch}], Loss: {train_loss:.6f}")

    # Component-wise Evaluation
    COMPONENTS = {
        "mic1_fan": (os.path.join(LIGHT_DIR, "mic1_fan"), HEAVY_DIR),
        "mic2_fan_motor": (os.path.join(LIGHT_DIR, "mic2_fan_motor"), HEAVY_DIR),
        "mic3_compressor": (os.path.join(LIGHT_DIR, "mic3_compressor"), HEAVY_DIR),
        "mic4_small_fan": (os.path.join(LIGHT_DIR, "mic4_small_fan"), HEAVY_DIR),
        "mic5_oil_pump": (os.path.join(LIGHT_DIR, "mic5_oil_pump"), HEAVY_DIR)
    }

    for component, (normal_path, abnormal_path) in COMPONENTS.items():
        print(f"\nEvaluating {component}...")
        threshold = 0.01     # Set anomaly detection threshold
        # 0.01  
        # np.percentile(train_loss, 95)  # 95th percentile of training losses
        scores, decisions, y_true, eval_time = evaluate_component(model, normal_path, abnormal_path, threshold)
        save_results(component, scores, decisions, y_true)

if __name__ == "__main__":
    main()


