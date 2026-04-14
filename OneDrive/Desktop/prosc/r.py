import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

# ==========================================
# 1. DATA LOADING & SPLITTING (6:2:2)
# ==========================================
print("--- Initializing Data Pipeline ---")
try:
    # Load train/val/test from separate files
    train_archive = np.load('train.npz')
    val_archive = np.load('val.npz')
    test_archive = np.load('test.npz')

    X_train, Y_train = train_archive['x'].astype('float32'), train_archive['y'].astype('float32')
    X_val, Y_val = val_archive['x'].astype('float32'), val_archive['y'].astype('float32')
    X_test, Y_test = test_archive['x'].astype('float32'), test_archive['y'].astype('float32')

    with open('adj_pems08.pkl', 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
    adj_mx = adj_data[2] if isinstance(adj_data, (list, tuple)) else adj_data

    print(f"Loaded train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}, Adj: {adj_mx.shape}")

    # --- Z-score Normalization ---
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    Y_train = (Y_train - mean) / std
    X_val = (X_val - mean) / std
    Y_val = (Y_val - mean) / std
    X_test = (X_test - mean) / std
    Y_test = (Y_test - mean) / std
    print(f"Data normalized: mean={mean:.2f}, std={std:.2f}")

    # --- Adjacency Matrix Normalization ---
    def normalize_adj(adj):
        D = np.sum(adj, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
        return D_inv_sqrt @ adj @ D_inv_sqrt
    adj_mx = normalize_adj(adj_mx)
    print("Adjacency matrix normalized.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()


# Creating Loaders
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), batch_size=64)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)), batch_size=64)

# ==========================================
# 2. DEFINE MODULES (Spatial & Temporal)
# ==========================================
class LightST(nn.Module):
    def __init__(self, adj, in_channels, out_channels):
        super(LightST, self).__init__()
        self.register_buffer('adj', torch.from_numpy(adj).float())
        
        # Spatial: Graph Convolution logic
        self.spatial_weight = nn.Parameter(torch.randn(in_channels, out_channels))
        # Temporal: Time logic (1D Conv over 12 steps)
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        
        self.fc_out = nn.Linear(out_channels, 1)

    def forward(self, x):
        # x: [Batch, Time, Nodes, Feature] -> [64, 12, 170, 1]
        
        # 1. Spatial Phase
        x = torch.matmul(x, self.spatial_weight)
        x = torch.matmul(self.adj, x) 
        
        # 2. Temporal Phase
        x = x.permute(0, 3, 1, 2) # [B, C, T, N]
        x = torch.relu(self.temporal_conv(x))
        
        # 3. Output Projection
        x = x.permute(0, 2, 3, 1) # [B, T, N, C]
        return self.fc_out(x)

    # --- 1. LIGHTST MODULE (Member 1) ---
    class LightST_Model(nn.Module):
        def __init__(self, adj_matrix):
            super().__init__()
            self.adj = torch.from_numpy(adj_matrix).float()
            self.conv = nn.Conv2d(1, 32, kernel_size=(3,1), padding=(1,0))
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = torch.relu(self.conv(x))
            x = x.permute(0, 2, 3, 1)
            return self.fc(x)

    # --- 2. DIGITAL TWIN & VRP MODULE (Member 2) ---
    class UrbanFreightSystem:
        def __init__(self, adj_mx):
            self.digital_twin_env = adj_mx
        def calculate_carbon_vrp(self, predictions):
            avg_predicted_flow = predictions.mean(axis=1)
            carbon_emissions = []
            for step in range(12):
                flow_penalty = 1 + (avg_predicted_flow[step] / 1000)
                ce = 5.2 * 110 * flow_penalty
                carbon_emissions.append(ce.item())
            return carbon_emissions

    # --- 3. 12-GRAPH PDF GENERATOR (Base Paper Comparison) ---
    from matplotlib.backends.backend_pdf import PdfPages
    def generate_research_pdf(my_mae, my_rmse, my_mape, my_dist, my_carbon, my_route):
        with PdfPages('Urban_Freight_Research_Results.pdf') as pdf:
            steps = np.arange(1, 13)
            baselines = {
                'GraphWaveNet': [13.2, 13.8, 14.5, 15.1, 15.6, 16.2, 16.8, 17.3, 17.8, 18.2, 18.7, 19.1],
                'AGCRN': [15.5, 16.1, 16.8, 17.4, 18.1, 18.8, 19.5, 20.2, 20.9, 21.6, 22.3, 23.0],
                'PDFormer': [12.1, 12.6, 13.1, 13.5, 13.9, 14.3, 14.7, 15.1, 15.5, 15.9, 16.3, 16.7]
            }
            metrics = [('MAE', my_mae), ('RMSE', my_rmse), ('MAPE', my_mape), ('Distance', my_dist), ('Carbon', my_carbon), ('Route', my_route)]
            for metric_name, data in metrics:
                plt.figure(figsize=(8, 5))
                if metric_name in ['MAE', 'RMSE', 'MAPE']:
                    plt.plot(steps, baselines['GraphWaveNet'], label='GraphWaveNet', linestyle='--')
                    plt.plot(steps, baselines['AGCRN'], label='AGCRN', linestyle='--')
                    plt.plot(steps, baselines['PDFormer'], label='PDFormer', linestyle='--')
                plt.plot(steps, data, label='Our LightST-VRP', color='black', marker='s', linewidth=2)
                plt.title(f"{metric_name} Comparison")
                plt.xlabel("Forecasting Step (5-60 min)")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

# ==========================================
# 3. TRAINING SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightST(adj_mx, 1, 32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.L1Loss() # MAE Loss as per paper standards



# ==========================================
# 4. EXECUTION LOOP
# ==========================================
print(f"\nStarting Training on {device}...")
for epoch in range(1, 9): # 8 Epochs for speed
    model.train()
    total_loss = 0
    for inputs, targets in DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=256, shuffle=True):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch}/8 | Train Loss (MAE): {total_loss/len(train_loader):.4f}")
print("\n--- Model Training Finished ---")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def masked_mape(y_true, y_pred, threshold=0.1):
    mask = y_true > threshold
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- Evaluation on Test Set ---
model.eval()
y_true_all = []
y_pred_all = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_true_all.append(targets.numpy())
        y_pred_all.append(outputs.cpu().numpy())

y_true = np.concatenate(y_true_all, axis=0).flatten()
y_pred = np.concatenate(y_pred_all, axis=0).flatten()

# Calculate Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = masked_mape(y_true, y_pred)

print(f"\n--- Test Metrics ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")

# --- Visualization: Metric Comparison Graph ---

# --- Step-wise Metrics and Plots (Paper Style) ---
def calculate_metrics_per_step(y_true, y_pred):
    # y_true/y_pred shape: [Samples, 12, Sensors, 1]
    mae_steps = []
    rmse_steps = []
    mape_steps = []
    for t in range(12):
        true = y_true[:, t, :, :].flatten()
        pred = y_pred[:, t, :, :].flatten()
        mae_t = np.mean(np.abs(true - pred))
        rmse_t = np.sqrt(np.mean((true - pred)**2))
        mask = true > 0.1
        mape_t = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
        mae_steps.append(mae_t)
        rmse_steps.append(rmse_t)
        mape_steps.append(mape_t)

    return mae_steps, mape_steps, rmse_steps

# --- Save step-wise metrics to CSV ---
import csv
y_true_sw = np.concatenate(y_true_all, axis=0)
y_pred_sw = np.concatenate(y_pred_all, axis=0)
mae_s, mape_s, rmse_s = calculate_metrics_per_step(y_true_sw, y_pred_sw)
steps = np.arange(1, 13)
csv_filename = 'lightst_stepwise_metrics.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Step', 'MAE', 'MAPE', 'RMSE'])
    for i in range(12):
        writer.writerow([steps[i], mae_s[i], mape_s[i], rmse_s[i]])
print(f"Step-wise metrics saved to {csv_filename}")

# --- Visualization: Predicted vs Actual (Time Series) ---
plt.figure(figsize=(12, 6))
# Plotting a small window of the test set for clarity
plt.plot(y_true[:288], label='Actual Flow', alpha=0.7)
plt.plot(y_pred[:288], label='Predicted Flow', linestyle='--')
plt.title('Traffic Flow Prediction: Actual vs Predicted (Next 24 Hours)')
plt.legend()
plt.show()