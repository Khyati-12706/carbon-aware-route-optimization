# Scientific Comparison Setup for Workflow Modules

## 1️⃣ Freight Demand & Traffic Prediction
- **Your Model:** LightST
- **Baselines:** STGCN, DCRNN, ASTGCN
- **Why Compare:**
  - STGCN: Standard benchmark, uses graph convolution + temporal CNN
  - DCRNN: Strong spatio-temporal model, uses RNN + graph diffusion
  - ASTGCN: Uses attention, captures daily/weekly patterns
- **Metrics:** MAE, RMSE, MAPE, Training time, Prediction accuracy
- **Graphs:**
  - MAE Comparison
  - RMSE Comparison
  - MAPE Comparison
  - Training Time
  - Prediction Accuracy vs Time Horizon

## 2️⃣ Dynamic Traffic Modeling
- **Your Model:** Deep Spatio-Temporal Dynamic Traffic Model
- **Baselines:** T-GCN, ST-ResNet, ConvLSTM
- **Metrics:** Traffic prediction RMSE, Congestion prediction accuracy, Computation time

## 3️⃣ Vehicle Routing Optimization
- **Your Model:** Temporal & Spatial Attention RL VRP
- **Baselines:** Genetic Algorithm, Ant Colony Optimization, Deep Q Network
- **Metrics:** Route distance, Fuel consumption, Delivery time, Convergence speed

## 4️⃣ Digital Twin Optimization
- **Your Model:** Adaptive Digital Twin Framework
- **Baselines:** Edge Computing Offloading, Cloud-based Traffic Simulation, Multi-access Edge Computing
- **Metrics:** Latency, Migration cost, System response time

## 5️⃣ Carbon Emission Estimation
- **Your Model:** RSML-TUSCE
- **Baselines:** MOVES, COPERT, VT-Micro
- **Metrics:** CO₂ emission prediction error, Total carbon emission, Energy efficiency

---

**Note:**
- For each module, compare your model with the most relevant, well-known baselines.
- Use standard metrics for fair benchmarking.
- Present results in clear, publication-ready graphs.
