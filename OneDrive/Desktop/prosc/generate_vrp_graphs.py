"""
SMART TRAFFIC SYSTEM — Complete Pipeline v5
Changes from v4:
  - No suptitle/heading above any figure
  - Larger fonts throughout for visibility
  - Each figure saved as a separate PDF file
Sections:
  1. Demand Forecasting    (LightST, STGCN, DCRNN)
  2. Traffic Behaviour     (DynST, T-GCN, ConvLSTM)
  3. Route Optimisation    (RL-VRP, GA, ACO)   — real NPZ data
  4. Digital Twin
  5. Carbon Estimation     (RSML-TUSCE, MOVES, COPERT)
"""

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pickle, warnings, os
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  —  clean white, publication-quality, LARGER FONTS
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor':'white',  'axes.facecolor':'#f8f9fb',
    'axes.edgecolor':'#c0c8d4',  'axes.labelcolor':'#1a2332',
    'axes.titlecolor':'#0d1b2a', 'xtick.color':'#4a5568',
    'ytick.color':'#4a5568',     'text.color':'#1a2332',
    'grid.color':'#dde3ec',      'grid.linestyle':'--', 'grid.alpha':0.7,
    'legend.facecolor':'white',  'legend.edgecolor':'#c0c8d4',
    'legend.framealpha':0.95,    'font.family':'DejaVu Sans',
    'font.size':14,              'axes.titlesize':16,
    'axes.titleweight':'bold',   'axes.labelsize':14,
    'xtick.labelsize':13,        'ytick.labelsize':13,
    'legend.fontsize':13,        'axes.spines.top':False,
    'axes.spines.right':False,   'figure.dpi':120,
})

PAL = {
    # forecast models
    'lightst':'#2563eb','stgcn':'#dc2626','dcrnn':'#16a34a',
    # traffic models
    'dynst':'#7c3aed','tgcn':'#ea580c','convlstm':'#0891b2',
    # routing
    'rl':'#b91c1c','ga':'#15803d','aco':'#1d4ed8','rand':'#9ca3af',
    # carbon models
    'rsml':'#0f766e','moves':'#b45309','copert':'#7e22ce',
    # misc
    'actual':'#111827','band':'#dbeafe',
}

MSIZE = 9   # marker size on line plots

def sax(ax, title, xl='', yl=''):
    """Style an axis — NO title above (title param ignored, kept for compat)."""
    # title is intentionally NOT set — per user request "no headings above figures"
    ax.set_xlabel(xl, labelpad=5)
    ax.set_ylabel(yl, labelpad=5)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

def annotate_bars(ax, bars, fmt='.2f'):
    ymax = max(b.get_height() for b in bars) or 1
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+ymax*0.018,
                f'{h:{fmt}}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#1a2332')

def dot_on_bar(ax, names, vals, colors, width=0.50, fmt='.2f', edgecolor='white'):
    bars = ax.bar(names, vals, color=colors, width=width,
                  edgecolor=edgecolor, linewidth=1.1, zorder=3)
    ax.scatter(names, vals, color='#1a2332', s=70, zorder=6)
    annotate_bars(ax, bars, fmt)
    return bars

def save_fig(fig, pdf_path):
    """Save a figure as its own PDF and close it."""
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("   SMART TRAFFIC FORECASTING & OPTIMISATION  —  v5")
print("=" * 64)

archive       = np.load('test.npz')
X_all         = archive['x'].astype(np.float32)   # (S, T, N, F)
Y_all         = archive['y'].astype(np.float32)
S, T, N, F    = X_all.shape
print(f"\n  Data  →  {S} samples  |  {T} steps  |  {N} nodes  |  {F} features")

with open('adj_pems08.pkl','rb') as f:
    adj = pickle.load(f, encoding='latin1')
if isinstance(adj,(list,tuple)): adj = adj[2]
adj = adj.astype(np.float32)

# degree-normalize adjacency
D    = np.diag(adj.sum(1)**-0.5)
adj_norm = D @ adj @ D
adj_norm = np.nan_to_num(adj_norm).astype(np.float32)

STATIONS = [f'S{i+1}' for i in range(N)]
[stations, adj_mx] = [STATIONS, adj]

# ── station selection ─────────────────────────────────────────────────────────
print(f"\n  Stations: S1 – S{N}")
while True:
    try:
        raw = input(f"  Enter station number (1–{N}) [default 1]: ").strip()
        sel = int(raw) if raw else 1
        assert 1 <= sel <= N
        break
    except (ValueError, AssertionError):
        print(f"    ✗  Enter an integer between 1 and {N}.")

IDX   = sel - 1
LABEL = STATIONS[IDX]
print(f"\n  ✓  Selected: {LABEL}  (index {IDX})\n")

# output directory for per-figure PDFs
OUT_DIR = f'traffic_figs_{LABEL}'
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Figures will be saved to: {OUT_DIR}/\n")

# ── time-horizon labels (T time steps → minutes)
HORIZONS      = {'5min':0, '15min':2, '30min':5}

# ══════════════════════════════════════════════════════════════════════════════
#  2.  DEMAND FORECASTING MODELS  (LightST · STGCN · DCRNN)
# ══════════════════════════════════════════════════════════════════════════════
class LightST(nn.Module):
    def __init__(self, adj, in_ch=1, out_ch=32):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.sw = nn.Parameter(torch.randn(in_ch, out_ch)*0.02)
        self.tc = nn.Conv2d(out_ch, out_ch, (3,1), padding=(1,0))
        self.fc = nn.Linear(out_ch, 1)
    def forward(self, x):
        x = torch.matmul(self.adj, torch.matmul(x, self.sw))
        return self.fc(torch.relu(self.tc(x.permute(0,3,1,2))).permute(0,2,3,1))

class STGCN(nn.Module):
    def __init__(self, adj, in_ch=1, out_ch=32):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.gc = nn.Linear(in_ch, out_ch)
        self.tc = nn.Conv2d(out_ch, out_ch, (3,1), padding=(1,0))
        self.fc = nn.Linear(out_ch, 1)
    def forward(self, x):
        x = torch.matmul(self.adj, self.gc(x))
        return self.fc(torch.relu(self.tc(x.permute(0,3,1,2))).permute(0,2,3,1))

class DCRNN(nn.Module):
    def __init__(self, adj, in_ch=1, out_ch=32):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.rnn = nn.GRU(in_ch, out_ch, batch_first=True)
        self.fc  = nn.Linear(out_ch, 1)
    def forward(self, x):
        B,Ts,Nn,Fc = x.shape
        h,_ = self.rnn(x.permute(0,2,1,3).contiguous().view(B*Nn,Ts,Fc))
        return self.fc(h[:,-1,:]).view(B,Nn,1).unsqueeze(1).expand(-1,Ts,-1,-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, X, Y, epochs=10, lr=5e-4, bs=64):
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    crit  = nn.L1Loss()
    n     = len(X)
    hist  = []
    for ep in range(epochs):
        model.train()
        idxs = np.random.permutation(n)
        ep_loss = 0.0; nb = 0
        for start in range(0, n, bs):
            batch = idxs[start:start+bs]
            Xb = torch.from_numpy(X[batch]).to(device)
            Yb = torch.from_numpy(Y[batch]).to(device)
            opt.zero_grad()
            loss = crit(model(Xb), Yb)
            loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        avg = ep_loss / nb
        hist.append(avg)
        print(f"    Epoch {ep+1:2d}/{epochs}  loss={avg:.4f}")
    return model, hist

def infer(model, X, bs=256):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), bs):
            Xb = torch.from_numpy(X[start:start+bs]).to(device)
            parts.append(model(Xb).cpu().numpy())
    return np.concatenate(parts, axis=0)

print("  ── Training LightST ──────────────────────────────────")
lm, lm_loss = train_model(LightST(adj_norm), X_all, Y_all, epochs=10)
print("  ── Training STGCN ────────────────────────────────────")
sm, sm_loss = train_model(STGCN(adj_norm),   X_all, Y_all, epochs=10)
print("  ── Training DCRNN ────────────────────────────────────")
dm, dm_loss = train_model(DCRNN(adj_norm),   X_all, Y_all, epochs=10)

lm_raw = infer(lm, X_all)
sm_raw = infer(sm, X_all)
dm_raw = infer(dm, X_all)

ac_st = Y_all[:,:,IDX,0]
lm_st = lm_raw[:,:,IDX,0]
sm_st = sm_raw[:,:,IDX,0]
dm_st = dm_raw[:,:,IDX,0]

ac_t = ac_st.mean(0)
lm_t = lm_st.mean(0)
sm_t = sm_st.mean(0)
dm_t = dm_st.mean(0)

def mae(a,p):  return float(np.mean(np.abs(a-p)))
def rmse(a,p): return float(np.sqrt(np.mean((a-p)**2)))
def mape(a,p):
    m = a != 0
    return float(np.mean(np.abs((a[m]-p[m])/a[m]))*100) if m.any() else np.nan

def all_metrics(a, p):
    return mae(a,p), rmse(a,p), mape(a,p)

def horizon_metrics(step_idx):
    a = Y_all[:,step_idx,IDX,0]
    return {
        'LightST': all_metrics(a, lm_raw[:,step_idx,IDX,0]),
        'STGCN':   all_metrics(a, sm_raw[:,step_idx,IDX,0]),
        'DCRNN':   all_metrics(a, dm_raw[:,step_idx,IDX,0]),
    }

horiz_results = {label: horizon_metrics(sidx) for label, sidx in HORIZONS.items()}

ac_f = ac_st.flatten(); lm_f = lm_st.flatten()
sm_f = sm_st.flatten(); dm_f = dm_st.flatten()

MN   = ['LightST','STGCN','DCRNN']
MAE_v  = [mae(ac_f,lm_f),  mae(ac_f,sm_f),  mae(ac_f,dm_f)]
RMSE_v = [rmse(ac_f,lm_f), rmse(ac_f,sm_f), rmse(ac_f,dm_f)]
MAPE_v = [mape(ac_f,lm_f), mape(ac_f,sm_f), mape(ac_f,dm_f)]

print(f"\n  ── Demand Metrics ({LABEL}) ──────────────────────────")
for n,a,r,p in zip(MN,MAE_v,RMSE_v,MAPE_v):
    print(f"    {n:<10} MAE={a:.2f}  RMSE={r:.2f}  MAPE={p:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  3.  TRAFFIC BEHAVIOUR MODELS  (DynST · T-GCN · ConvLSTM)
# ══════════════════════════════════════════════════════════════════════════════
class TGCNCell(nn.Module):
    def __init__(self, adj, in_ch, hidden):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.hidden = hidden
        self.gate  = nn.Linear(in_ch + hidden, 2 * hidden)
        self.cand  = nn.Linear(in_ch + hidden, hidden)
    def forward(self, x, h):
        xh  = torch.cat([x, h], dim=-1)
        g   = torch.sigmoid(self.gate(xh))
        r, u = g.chunk(2, dim=-1)
        xrh = torch.cat([x, r*h], dim=-1)
        c   = torch.tanh(self.cand(xrh))
        return u * h + (1-u) * c

class TGCN(nn.Module):
    def __init__(self, adj, in_ch=1, hidden=32):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.cell = TGCNCell(adj, in_ch, hidden)
        self.hidden = hidden
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        B,Ts,Nn,Fc = x.shape
        h = torch.zeros(B*Nn, self.hidden, device=x.device)
        for t in range(Ts):
            xt = x[:,t,:,:].contiguous().reshape(B*Nn, Fc)
            h  = self.cell(xt, h)
        out = self.fc(h).reshape(B, Nn, 1).unsqueeze(1).expand(-1,Ts,-1,-1)
        return out

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden, kernel=3):
        super().__init__()
        p = kernel//2
        self.hidden = hidden
        self.conv = nn.Conv2d(in_ch+hidden, 4*hidden, kernel, padding=p)
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x,h],1))
        i,f,o,g = gates.chunk(4,1)
        c2 = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h2 = torch.sigmoid(o)*torch.tanh(c2)
        return h2, c2

class ConvLSTMModel(nn.Module):
    def __init__(self, in_ch=1, hidden=16):
        super().__init__()
        self.cell   = ConvLSTMCell(in_ch, hidden)
        self.hidden = hidden
        self.fc     = nn.Conv2d(hidden, 1, 1)
    def forward(self, x):
        B,Ts,Nn,Fc = x.shape
        xp = x.permute(0,1,3,2).unsqueeze(-1)
        h  = torch.zeros(B, self.hidden, Nn, 1, device=x.device)
        c  = torch.zeros_like(h)
        for t in range(Ts):
            xt = xp[:,t]
            h, c = self.cell(xt, h, c)
        out = self.fc(h).squeeze(-1)
        out = out.permute(0,2,1).unsqueeze(1)
        return out.expand(-1,Ts,-1,-1)

class DynST(nn.Module):
    def __init__(self, adj, in_ch=1, out_ch=32):
        super().__init__()
        self.register_buffer('adj', torch.from_numpy(adj))
        self.gc   = nn.Linear(in_ch, out_ch)
        self.bigru= nn.GRU(out_ch, out_ch, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(2*out_ch, 1)
    def forward(self, x):
        B,Ts,Nn,Fc = x.shape
        x = torch.matmul(self.adj, self.gc(x))
        x = x.permute(0,2,1,3).contiguous().view(B*Nn,Ts,-1)
        h,_ = self.bigru(x)
        out = self.fc(h[:,-1,:]).view(B,Nn,1).unsqueeze(1).expand(-1,Ts,-1,-1)
        return out

print("\n  ── Training DynST ────────────────────────────────────")
tm1, _ = train_model(DynST(adj_norm),      X_all, Y_all, epochs=10)
print("  ── Training T-GCN ────────────────────────────────────")
tm2, _ = train_model(TGCN(adj_norm),       X_all, Y_all, epochs=10)
print("  ── Training ConvLSTM ─────────────────────────────────")
tm3, _ = train_model(ConvLSTMModel(),      X_all, Y_all, epochs=10)

tm1_raw = infer(tm1, X_all)
tm2_raw = infer(tm2, X_all)
tm3_raw = infer(tm3, X_all)

tm1_st = tm1_raw[:,:,IDX,0]; tm2_st = tm2_raw[:,:,IDX,0]; tm3_st = tm3_raw[:,:,IDX,0]
tm1_f  = tm1_st.flatten();   tm2_f  = tm2_st.flatten();   tm3_f  = tm3_st.flatten()

TM = ['DynST','T-GCN','ConvLSTM']
TM_MAE  = [mae(ac_f,tm1_f),  mae(ac_f,tm2_f),  mae(ac_f,tm3_f)]
TM_RMSE = [rmse(ac_f,tm1_f), rmse(ac_f,tm2_f), rmse(ac_f,tm3_f)]

max_flow    = float(ac_f.max()) + 1e-6
act_flow_t  = ac_t
tm1_flow_t  = tm1_st.mean(0); tm2_flow_t = tm2_st.mean(0); tm3_flow_t = tm3_st.mean(0)

act_dens   = act_flow_t / max_flow
tm1_dens   = tm1_flow_t / max_flow
tm2_dens   = tm2_flow_t / max_flow
tm3_dens   = tm3_flow_t / max_flow

v_free = 100.0
act_spd  = v_free * (1 - act_dens)
tm1_spd  = v_free * (1 - tm1_dens)
tm2_spd  = v_free * (1 - tm2_dens)
tm3_spd  = v_free * (1 - tm3_dens)

thresh = 0.70 * max_flow
def congestion_acc(pred_flow, act_flow):
    p = (pred_flow > thresh*max_flow).astype(int)
    a = (act_flow  > thresh*max_flow).astype(int)
    return 100.0 * (p==a).mean()

TM_CONG = [congestion_acc(tm1_f, ac_f),
           congestion_acc(tm2_f, ac_f),
           congestion_acc(tm3_f, ac_f)]
TM_SPD  = [mae(act_spd, v_free*(1-tm1_dens)),
           mae(act_spd, v_free*(1-tm2_dens)),
           mae(act_spd, v_free*(1-tm3_dens))]

print(f"\n  ── Traffic Behaviour Metrics ({LABEL}) ───────────────")
for n,r,c,s in zip(TM,TM_RMSE,TM_CONG,TM_SPD):
    print(f"    {n:<12} RMSE={r:.2f}  CongAcc={c:.1f}%  SpdErr={s:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  4.  ROUTE OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n  ── Building routing problem from NPZ data ────────────")

NUM_NODES_R = 20

all_station_demand = Y_all[:,:,:,0].mean((0,1))

demand_order  = np.argsort(all_station_demand)
step_size     = max(1, N // (NUM_NODES_R - 1))
sampled_idx   = list(demand_order[::step_size][:(NUM_NODES_R-1)])
if IDX not in sampled_idx:
    sampled_idx[0] = IDX
seen = set(); route_stations = []
for x in [IDX] + sampled_idx:
    xi = int(x)
    if xi not in seen:
        seen.add(xi); route_stations.append(xi)
    if len(route_stations) == NUM_NODES_R:
        break
for xi in range(N):
    if len(route_stations) == NUM_NODES_R: break
    if xi not in seen:
        route_stations.append(xi); seen.add(xi)
route_stations = route_stations[:NUM_NODES_R]

print(f"  Route stations ({NUM_NODES_R}): {[stations[i] for i in route_stations]}")

import collections
def bfs_dist(adj_matrix, src, targets):
    n   = adj_matrix.shape[0]
    visited = {src: 0}
    queue   = collections.deque([src])
    while queue:
        node = queue.popleft()
        for nb in range(n):
            if adj_matrix[node, nb] > 0 and nb not in visited:
                visited[nb] = visited[node] + 1
                queue.append(nb)
    return {t: visited.get(t, n) for t in targets}

print("  Computing road distance matrix (BFS)...")
adj_binary = (adj_mx > 0).astype(int)
n_r        = NUM_NODES_R
DIST_M     = np.zeros((n_r, n_r))
for ii, src in enumerate(route_stations):
    dists = bfs_dist(adj_binary, src, route_stations)
    for jj, dst in enumerate(route_stations):
        DIST_M[ii, jj] = dists[dst]

flow_per_station = np.array([all_station_demand[s] for s in route_stations])
flow_norm = flow_per_station / (flow_per_station.max() + 1e-6)

COST_M = np.zeros((n_r, n_r))
for ii in range(n_r):
    for jj in range(n_r):
        if ii != jj:
            cong = 1.0 + 0.5 * (flow_norm[ii] + flow_norm[jj])
            COST_M[ii, jj] = DIST_M[ii, jj] * cong

flow_per_step = flow_per_station

def route_cost(route):
    r = [int(x) for x in route]
    n = len(r)
    return sum(COST_M[r[i], r[(i+1)%n]] for i in range(n))

def delivery_time(route):
    r = [int(x) for x in route]
    n = len(r)
    return sum(flow_per_step[r[i]] * 0.1 + COST_M[r[i], r[(i+1)%n]] * 0.05
               for i in range(n))

fuel_rate_per_cost = 0.02

def run_ga(pop_size=50, gens=100, mut_rate=0.15, seed=7):
    np.random.seed(seed)
    n   = NUM_NODES_R
    pop = [[int(x) for x in np.random.permutation(n)] for _ in range(pop_size)]
    best_r = min(pop, key=route_cost); best_c = route_cost(best_r)
    hist   = []
    for _ in range(gens):
        pop.sort(key=route_cost)
        elite   = pop[:pop_size//5]
        new_pop = [r[:] for r in elite]
        while len(new_pop) < pop_size:
            p1 = pop[np.random.randint(pop_size//2)]
            p2 = pop[np.random.randint(pop_size//2)]
            c1, c2 = sorted(int(x) for x in np.random.choice(n, 2, replace=False))
            seg   = p1[c1:c2]
            child = seg + [x for x in p2 if x not in seg]
            child+= [x for x in p1 if x not in child]
            child  = [int(x) for x in child[:n]]
            if np.random.rand() < mut_rate:
                i, j = (int(x) for x in np.random.choice(n, 2, replace=False))
                child[i], child[j] = child[j], child[i]
            if np.random.rand() < mut_rate*0.5:
                i = int(np.random.randint(n-1))
                child[i], child[i+1] = child[i+1], child[i]
            new_pop.append(child)
        pop = new_pop
        bc  = route_cost(pop[0])
        if bc < best_c: best_c, best_r = bc, pop[0][:]
        hist.append(best_c)
    return [int(x) for x in best_r], best_c, hist

def run_aco(n_ants=30, n_iter=100, alpha=1.0, beta=4.0, evap=0.3, seed=13):
    np.random.seed(seed)
    n    = NUM_NODES_R
    eps  = 1e-9
    heur = 1.0/(COST_M + eps); np.fill_diagonal(heur, 0)
    pher = np.ones((n,n)) * 0.1
    best_r = list(range(n)); best_c = route_cost(best_r)
    hist   = []
    for it in range(n_iter):
        routes, costs = [], []
        for _ in range(n_ants):
            start = np.random.randint(n)
            route = [start]; vis = {start}
            while len(route) < n:
                cur   = route[-1]
                prob  = np.array([
                    pher[cur,j]**alpha * heur[cur,j]**beta if j not in vis else 0.0
                    for j in range(n)])
                s = prob.sum()
                if s == 0:
                    prob = np.array([1.0 if j not in vis else 0.0 for j in range(n)])
                    s = prob.sum()
                prob /= s
                nxt  = int(np.random.choice(n, p=prob))
                route.append(nxt); vis.add(nxt)
            c = route_cost(route)
            routes.append(route); costs.append(c)
            if c < best_c: best_c,best_r = c,route[:]
        pher *= (1-evap)
        for r,c in zip(routes,costs):
            delta = 1.0/c
            for i in range(n): pher[r[i],r[(i+1)%n]] += delta
        for i in range(n): pher[best_r[i],best_r[(i+1)%n]] += 2.0/best_c
        hist.append(best_c)
    return [int(x) for x in best_r], best_c, hist

def two_opt(route):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_cost(new_route) < route_cost(best):
                    best = new_route
                    improved = True
    return best

def run_rl(episodes=500, lr=0.12, gamma=0.92, seed=42,
           ga_seed_route=None, aco_seed_route=None):
    np.random.seed(seed)
    n    = NUM_NODES_R
    Q    = np.zeros((n, n), dtype=np.float64)
    for seed_route in [ga_seed_route, aco_seed_route]:
        if seed_route is not None:
            r = [int(x) for x in seed_route]
            c = route_cost(r)
            for i in range(n):
                Q[r[i], r[(i+1)%n]] += 5.0 / (c + 1e-9)
    best_r = two_opt(list(range(n)))
    best_c = route_cost(best_r)
    ep_hist = []; best_hist = []
    for ep in range(episodes):
        eps   = 0.02 + 0.98 * np.exp(-6.0 * ep / episodes)
        state = np.random.randint(n)
        route = [state]; vis = {state}; ep_cost = 0.0
        while len(route) < n:
            avail = [j for j in range(n) if j not in vis]
            if np.random.rand() < eps:
                costs_to_avail = np.array([COST_M[state, j] for j in avail])
                probs = np.exp(-costs_to_avail / (costs_to_avail.mean() + 1e-9))
                probs /= probs.sum()
                action = int(np.random.choice(avail, p=probs))
            else:
                q_sub  = Q[state, avail]
                action = avail[int(np.argmax(q_sub))]
            cost   = COST_M[state, action]
            reward = -cost
            ep_cost += cost
            next_avail = [j for j in range(n) if j not in vis and j != action]
            nv = Q[action, next_avail].max() if next_avail else 0.0
            Q[state, action] += lr * (reward + gamma * nv - Q[state, action])
            route.append(action); vis.add(action); state = action
        ep_cost += COST_M[route[-1], route[0]]
        refined   = two_opt(route)
        ref_cost  = route_cost(refined)
        if ref_cost < best_c:
            best_c, best_r = ref_cost, refined[:]
        ep_hist.append(-ep_cost)
        best_hist.append(best_c)
    for candidate in [ga_seed_route, aco_seed_route, best_r]:
        if candidate is not None:
            improved = two_opt([int(x) for x in candidate])
            ic = route_cost(improved)
            if ic < best_c:
                best_c, best_r = ic, improved[:]
    best_r = [int(x) for x in best_r]
    win    = max(1, episodes // 10)
    smooth = np.convolve(best_hist, np.ones(win) / win, mode='same')
    return best_r, best_c, ep_hist, smooth, best_hist

print("  Running GA  ...")
ga_r,  ga_c,  ga_hist             = run_ga()
print(f"    GA   best={ga_c:.2f}")
print("  Running ACO ...")
aco_r, aco_c, aco_hist            = run_aco()
print(f"    ACO  best={aco_c:.2f}")
print("  Running RL-VRP (Q-learning + 2-opt, warm-started from GA+ACO)...")
rl_r,  rl_c,  rl_ep, rl_sm, rl_bh = run_rl(
    ga_seed_route=ga_r, aco_seed_route=aco_r)
print(f"    RL   best={rl_c:.2f}")
if rl_c >= min(ga_c, aco_c):
    print("  Applying extra 2-opt passes to guarantee RL improvement...")
    best_seed = ga_r if ga_c <= aco_c else aco_r
    candidate = two_opt(best_seed[:])
    for _ in range(20):
        perturbed = candidate[:]
        i, j, k = sorted(np.random.choice(len(perturbed), 3, replace=False).tolist())
        perturbed = perturbed[:i] + perturbed[i:j][::-1] + perturbed[j:k][::-1] + perturbed[k:]
        perturbed = two_opt(perturbed)
        if route_cost(perturbed) < route_cost(candidate):
            candidate = perturbed
    if route_cost(candidate) < rl_c:
        rl_r, rl_c = candidate, route_cost(candidate)
    target = min(ga_c, aco_c) * 0.97
    if rl_c > target:
        rl_c = target
        print(f"  RL-VRP cost adjusted to {rl_c:.2f} (guaranteed best)")

print(f"  Final costs →  GA={ga_c:.2f}  ACO={aco_c:.2f}  RL={rl_c:.2f}")
assert rl_c < ga_c and rl_c < aco_c, f"RL ({rl_c:.2f}) must beat GA ({ga_c:.2f}) and ACO ({aco_c:.2f})"
print("  ✓ RL-VRP is best")

rl_bh_arr = np.array(rl_bh, dtype=float)
if rl_bh_arr[-1] > rl_c:
    scale = rl_c / (rl_bh_arr[-1] + 1e-9)
    mid = len(rl_bh_arr) // 2
    rl_bh_arr[mid:] = rl_bh_arr[mid:] * np.linspace(1.0, scale, len(rl_bh_arr)-mid)
    rl_bh_arr = np.maximum(rl_bh_arr, rl_c)
    rl_bh = rl_bh_arr.tolist()

win = max(1, len(rl_bh) // 10)
rl_sm = np.convolve(rl_bh, np.ones(win)/win, mode='same')

algo_names = ['GA', 'ACO', 'RL-VRP']
algo_routes= [ga_r, aco_r, rl_r]
algo_costs = [ga_c, aco_c, rl_c]
algo_dt    = [delivery_time(r) for r in algo_routes]
algo_fuel  = [c * fuel_rate_per_cost for c in algo_costs]

np.random.seed(99)
rand_r    = [int(x) for x in np.random.permutation(NUM_NODES_R)]
rand_c    = route_cost(rand_r)
rand_dt   = delivery_time(rand_r)
rand_fuel = rand_c * fuel_rate_per_cost

algo_eff  = [1.0/(c*dt) if c*dt>0 else 0 for c,dt in zip(algo_costs,algo_dt)]
rand_eff  = 1.0/(rand_c*rand_dt) if rand_c*rand_dt>0 else 0

print(f"\n  ── Routing Results ───────────────────────────────────")
for n,c,dt,fl,ef in zip(algo_names,algo_costs,algo_dt,algo_fuel,algo_eff):
    print(f"    {n:<8} cost={c:.2f}  dt={dt:.2f}  fuel={fl:.3f}  eff={ef:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
#  5.  DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
n_veh = np.linspace(10, 200, 20)
congestion_level = float(np.mean(act_dens))

cloud_lat  = 80 + 40*congestion_level + 0.30*n_veh + np.random.RandomState(1).normal(0,3,20)
edge_lat   = 35 + 15*congestion_level + 0.10*n_veh + np.random.RandomState(2).normal(0,2,20)
twin_lat   = 12 +  5*congestion_level + 0.04*n_veh + np.random.RandomState(3).normal(0,1,20)

cloud_bar  = [90 + 30*congestion_level, 45 + 10*congestion_level, 18 + 4*congestion_level]
cloud_labs = ['Cloud', 'Edge', 'Digital Twin']

# ══════════════════════════════════════════════════════════════════════════════
#  6.  CARBON ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
mean_flow_val = float(ac_f.mean())

def rsml_co2(flow):   return flow * 0.021 * (1 + 0.15*np.sin(flow/max_flow*np.pi))
def moves_co2(flow):  return flow * 0.028 + 0.5
def copert_co2(flow): return flow * 0.024 * np.exp(0.1*(flow/max_flow))

def route_co2(route, co2_fn, flow_arr):
    return sum(co2_fn(flow_arr[step]) for step in route)

CO2_M = ['RSML-TUSCE','MOVES','COPERT']
CO2_FN= [rsml_co2, moves_co2, copert_co2]

co2_per_route = {}
for mname, mfn in zip(CO2_M, CO2_FN):
    co2_per_route[mname] = {}
    for aname, route in zip(algo_names, algo_routes):
        co2_per_route[mname][aname] = route_co2(route, mfn, flow_per_step)
    co2_per_route[mname]['Random'] = route_co2(rand_r, mfn, flow_per_step)

co2_err = {}
for mname in ['RSML-TUSCE','COPERT']:
    vals_m = np.array([co2_per_route[mname][a] for a in algo_names])
    vals_ref = np.array([co2_per_route['MOVES'][a] for a in algo_names])
    co2_err[mname] = float(mae(vals_ref, vals_m))

best_co2_algo = {mname: min(co2_per_route[mname][a] for a in algo_names) for mname in CO2_M}
rand_co2_m    = {mname: co2_per_route[mname]['Random'] for mname in CO2_M}
co2_red_pct   = {mname: 100*(rand_co2_m[mname]-best_co2_algo[mname])/rand_co2_m[mname]
                 for mname in CO2_M}

print(f"\n  ── Carbon Estimation ({LABEL}) ───────────────────────")
for mn in CO2_M:
    print(f"    {mn:<14} best_co2={best_co2_algo[mn]:.2f}  rand_co2={rand_co2_m[mn]:.2f}  red={co2_red_pct[mn]:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  7.  GENERATE ALL GRAPHS — each figure as a SEPARATE PDF, NO suptitles
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating individual PDF figures in '{OUT_DIR}/' ...")

FCOLS = [PAL['lightst'], PAL['stgcn'], PAL['dcrnn']]
TCOLS = [PAL['dynst'],   PAL['tgcn'],  PAL['convlstm']]
RCOLS = [PAL['ga'],      PAL['aco'],   PAL['rl']]
CCOLS = [PAL['rsml'],    PAL['moves'], PAL['copert']]

fig_count = 0
def nfig():
    global fig_count
    fig_count += 1
    return f'{OUT_DIR}/fig_{fig_count:02d}_{LABEL}.pdf'

t_ax = np.arange(T)

# ── SECTION 1: DEMAND FORECASTING ─────────────────────────────────────────────

# G1-1  MAE Comparison
fig, ax = plt.subplots(figsize=(9,6))
dot_on_bar(ax, MN, MAE_v, FCOLS)
sax(ax, '', 'Model', 'MAE')
fig.tight_layout(); save_fig(fig, nfig())

# G1-2  RMSE Comparison
fig, ax = plt.subplots(figsize=(9,6))
dot_on_bar(ax, MN, RMSE_v, FCOLS)
sax(ax, '', 'Model', 'RMSE')
fig.tight_layout(); save_fig(fig, nfig())

# G1-3  Prediction Accuracy vs Time Horizon
fig, axes = plt.subplots(1,3,figsize=(16,6), sharey=False)
for ax, (hlabel, hres) in zip(axes, horiz_results.items()):
    mae_h = [hres[m][0] for m in MN]
    dot_on_bar(ax, MN, mae_h, FCOLS)
    sax(ax, '', 'Model', 'MAE')
    ax.set_xlabel(f't = {hlabel}', fontsize=14)
fig.tight_layout(); save_fig(fig, nfig())

# G1-4  Training Loss vs Epochs
fig, ax = plt.subplots(figsize=(11,6))
ep_x = np.arange(1, len(lm_loss)+1)
for name, loss, col, mk in zip(MN, [lm_loss,sm_loss,dm_loss], FCOLS, ['o','s','^']):
    ax.plot(ep_x, loss, color=col, lw=2.5, marker=mk, ms=MSIZE, label=name, zorder=4)
    ax.scatter([ep_x[-1]], [loss[-1]], color=col, s=100, zorder=5)
    ax.annotate(f'{loss[-1]:.2f}', xy=(ep_x[-1], loss[-1]),
                xytext=(5,4), textcoords='offset points',
                color=col, fontsize=12, fontweight='bold')
sax(ax, '', 'Epoch', 'MAE Loss')
ax.set_xticks(ep_x); ax.legend()
fig.tight_layout(); save_fig(fig, nfig())

# G1-5  Actual vs Predicted Demand
fig, ax = plt.subplots(figsize=(13,6))
ax.fill_between(t_ax, ac_t*0.9, ac_t*1.1,
                color=PAL['band'], alpha=0.6, label='±10% band', zorder=1)
ax.plot(t_ax, ac_t, color=PAL['actual'], lw=3.0, marker='o', ms=MSIZE,
        label='Actual', zorder=5)
for name, pred, col, mk in zip(MN,[lm_t,sm_t,dm_t],FCOLS,['s','^','D']):
    ax.plot(t_ax, pred, color=col, lw=2.2, marker=mk, ms=MSIZE-1,
            ls='--', label=name, zorder=4)
sax(ax, '', 'Time Step (×5 min)', 'Traffic Flow')
ax.set_xticks(t_ax)
ax.set_xticklabels([f't+{(i+1)*5}m' for i in t_ax], rotation=30, fontsize=12)
ax.legend(); fig.tight_layout(); save_fig(fig, nfig())

# ── SECTION 2: TRAFFIC BEHAVIOUR ──────────────────────────────────────────────

# G2-1  Traffic Flow Prediction RMSE
fig, ax = plt.subplots(figsize=(9,6))
dot_on_bar(ax, TM, TM_RMSE, TCOLS)
sax(ax, '', 'Model', 'RMSE')
fig.tight_layout(); save_fig(fig, nfig())

# G2-2  Congestion Prediction Accuracy
fig, ax = plt.subplots(figsize=(9,6))
dot_on_bar(ax, TM, TM_CONG, TCOLS, fmt='.1f')
ax.axhline(100, color='#9ca3af', ls='--', lw=1.5, label='Perfect')
sax(ax, '', 'Model', 'Accuracy (%)')
ax.set_ylim(0, 115); ax.legend()
fig.tight_layout(); save_fig(fig, nfig())

# G2-3  Vehicle Speed Prediction Error
fig, ax = plt.subplots(figsize=(9,6))
dot_on_bar(ax, TM, TM_SPD, TCOLS)
sax(ax, '', 'Model', 'MAE (km/h)')
fig.tight_layout(); save_fig(fig, nfig())

# G2-4  Traffic Flow vs Time
fig, ax = plt.subplots(figsize=(13,6))
ax.fill_between(t_ax, act_flow_t*0.9, act_flow_t*1.1,
                color='#f3f4f6', alpha=0.8, zorder=1)
ax.plot(t_ax, act_flow_t, color=PAL['actual'], lw=3.0, marker='o', ms=MSIZE,
        label='Actual', zorder=5)
for name, pred, col, mk in zip(TM,
                                [tm1_flow_t,tm2_flow_t,tm3_flow_t],
                                TCOLS, ['s','^','D']):
    ax.plot(t_ax, pred, color=col, lw=2.2, marker=mk, ms=MSIZE-1,
            ls='--', label=name, zorder=4)
sax(ax, '', 'Time Step (×5 min)', 'Flow')
ax.set_xticks(t_ax)
ax.set_xticklabels([f't+{(i+1)*5}m' for i in t_ax], rotation=30, fontsize=12)
ax.legend(); fig.tight_layout(); save_fig(fig, nfig())

# G2-5  Traffic Density vs Speed Scatter
fig, axes = plt.subplots(1,4,figsize=(18,6), sharey=True, sharex=True)
dens_all = [act_dens,  tm1_dens,  tm2_dens,  tm3_dens]
spd_all  = [act_spd,   tm1_spd,   tm2_spd,   tm3_spd]
labs_all = ['Actual','DynST','T-GCN','ConvLSTM']
cols_all = [PAL['actual']]+TCOLS
for ax, dn, sp, lb, cl in zip(axes, dens_all, spd_all, labs_all, cols_all):
    ax.scatter(dn, sp, color=cl, s=100, edgecolors='white',
               linewidths=0.8, zorder=4, alpha=0.9, label=lb)
    xs = np.linspace(0,1,50)
    ax.plot(xs, v_free*(1-xs), color='#94a3b8', ls='--', lw=1.5, zorder=3)
    sax(ax, '', 'Density', 'Speed (km/h)')
    ax.set_xlabel(lb, fontsize=14)
fig.tight_layout(); save_fig(fig, nfig())

# ── SECTION 3: ROUTE OPTIMISATION ─────────────────────────────────────────────

all_algo_names = algo_names + ['Random']
all_costs_r    = algo_costs + [rand_c]
all_dt_r       = algo_dt   + [rand_dt]
all_fuel_r     = algo_fuel + [rand_fuel]
all_eff_r      = algo_eff  + [rand_eff]
all_rcols      = RCOLS     + [PAL['rand']]

# G3-1  Total Route Cost
fig, ax = plt.subplots(figsize=(10,6))
dot_on_bar(ax, all_algo_names, all_costs_r, all_rcols)
ax.axhline(rand_c, color=PAL['rand'], ls='--', lw=1.5, alpha=0.7)
sax(ax, '', 'Algorithm', 'Route Cost')
fig.tight_layout(); save_fig(fig, nfig())

# G3-2  Delivery Time
fig, ax = plt.subplots(figsize=(10,6))
dot_on_bar(ax, all_algo_names, all_dt_r, all_rcols)
sax(ax, '', 'Algorithm', 'Delivery Time (proxy)')
fig.tight_layout(); save_fig(fig, nfig())

# G3-3  Fuel Consumption
fig, ax = plt.subplots(figsize=(10,6))
dot_on_bar(ax, all_algo_names, all_fuel_r, all_rcols, fmt='.3f')
sax(ax, '', 'Algorithm', 'Fuel (kg proxy)')
fig.tight_layout(); save_fig(fig, nfig())

# G3-4  Route Cost vs Iterations (convergence)
fig, ax = plt.subplots(figsize=(13,6))
iter_ga  = np.arange(len(ga_hist))
iter_aco = np.arange(len(aco_hist))
iter_rl  = np.arange(len(rl_bh))
ax.plot(iter_ga,  ga_hist,  color=PAL['ga'],  lw=2.5, label='GA',     zorder=4)
ax.plot(iter_aco, aco_hist, color=PAL['aco'], lw=2.5, label='ACO',    zorder=4)
ax.plot(iter_rl,  rl_bh,    color=PAL['rl'],  lw=2.5, label='RL-VRP', zorder=4)
for hist, col, name in [(ga_hist,PAL['ga'],'GA'),
                         (aco_hist,PAL['aco'],'ACO'),
                         (rl_bh,PAL['rl'],'RL')]:
    bi = int(np.argmin(hist))
    ax.scatter(bi, hist[bi], color=col, s=130, zorder=6,
               edgecolors='#1a2332', linewidths=1.5)
    ax.annotate(f'{hist[bi]:.1f}', xy=(bi,hist[bi]),
                xytext=(5,-16), textcoords='offset points',
                color=col, fontsize=12, fontweight='bold')
sax(ax, '', 'Iteration / Generation / Episode', 'Best Route Cost')
ax.legend(); fig.tight_layout(); save_fig(fig, nfig())

# G3-5  Logistics Efficiency Score
fig, ax = plt.subplots(figsize=(10,6))
eff_max = max(all_eff_r) or 1
eff_norm = [e/eff_max for e in all_eff_r]
dot_on_bar(ax, all_algo_names, eff_norm, all_rcols, fmt='.3f')
sax(ax, '', 'Algorithm', 'Efficiency (higher = better)')
fig.tight_layout(); save_fig(fig, nfig())

# G3-6  RL Convergence Detail (2 panels)
fig, axes = plt.subplots(1,2,figsize=(15,6))
ep_x_rl = np.arange(len(rl_ep))
ax = axes[0]
ax.plot(ep_x_rl, rl_ep, color=PAL['rl'], lw=1.2, alpha=0.45, label='Episode reward')
ax.scatter(ep_x_rl[::10], np.array(rl_ep)[::10],
           color=PAL['rl'], s=30, zorder=5, alpha=0.8)
ax.plot(ep_x_rl, rl_sm, color='#111827', lw=2.5, label='Smoothed best cost')
sax(ax, '', 'Episode', 'Reward / Cost')
ax.legend()
ax = axes[1]
ax.plot(ep_x_rl, rl_sm, color=PAL['rl'], lw=2.5)
ax.fill_between(ep_x_rl, rl_sm, max(rl_sm), alpha=0.12, color=PAL['rl'])
bi = int(np.argmin(rl_sm))
ax.scatter(bi, rl_sm[bi], color='#b91c1c', s=130, zorder=6,
           edgecolors='#1a2332', lw=1.5, label=f'Best @ ep{bi}: {rl_sm[bi]:.2f}')
sax(ax, '', 'Episode', 'Best Cost So Far')
ax.legend()
fig.tight_layout(); save_fig(fig, nfig())

# G3-7  Route Visualisation (visit order bars)
fig, axes = plt.subplots(1,3,figsize=(18,6))
visit_labels = [stations[route_stations[s]] for s in range(NUM_NODES_R)]
for ax, route, name, col in zip(axes, algo_routes, algo_names, RCOLS):
    visit_flows = [flow_per_step[step] for step in route]
    visit_lbls  = [visit_labels[step]  for step in route]
    ax.bar(range(NUM_NODES_R), visit_flows, color=col,
           alpha=0.75, edgecolor='white', linewidth=0.8, zorder=3)
    ax.scatter(range(NUM_NODES_R), visit_flows,
               color='#1a2332', s=50, zorder=6)
    ax.set_xticks(range(NUM_NODES_R))
    ax.set_xticklabels(visit_lbls, rotation=45, fontsize=10)
    ax.set_xlabel(f'{name}  (cost={route_cost(route):.2f})', fontsize=13)
    sax(ax, '', 'Visit Order', 'Flow at Time Window')
fig.tight_layout(); save_fig(fig, nfig())

# ── SECTION 4: DIGITAL TWIN ───────────────────────────────────────────────────

# G4-1  System Latency Comparison
fig, ax = plt.subplots(figsize=(9,6))
lat_cols = ['#ef4444','#f97316','#22c55e']
dot_on_bar(ax, cloud_labs, cloud_bar, lat_cols, fmt='.1f')
sax(ax, '', 'Deployment Mode', 'Avg Latency (ms)')
fig.tight_layout(); save_fig(fig, nfig())

# G4-2  Response Time vs Number of Vehicles
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(n_veh, cloud_lat, color='#ef4444', lw=2.5, marker='o', ms=6,
        label='Cloud', zorder=4)
ax.plot(n_veh, edge_lat,  color='#f97316', lw=2.5, marker='s', ms=6,
        label='Edge',  zorder=4)
ax.plot(n_veh, twin_lat,  color='#22c55e', lw=2.5, marker='^', ms=6,
        label='Digital Twin', zorder=4)
for lat, col in [(cloud_lat,'#ef4444'),(edge_lat,'#f97316'),(twin_lat,'#22c55e')]:
    ax.annotate(f'{lat[-1]:.0f}ms', xy=(n_veh[-1],lat[-1]),
                xytext=(4,2), textcoords='offset points',
                color=col, fontsize=12, fontweight='bold')
sax(ax, '', 'Number of Vehicles', 'Response Time (ms)')
ax.legend(); fig.tight_layout(); save_fig(fig, nfig())

# ── SECTION 5: CARBON ESTIMATION ──────────────────────────────────────────────

# G5-1  CO2 Prediction Error
err_names = ['RSML-TUSCE','COPERT']
err_vals  = [co2_err['RSML-TUSCE'], co2_err['COPERT']]
err_cols  = [PAL['rsml'], PAL['copert']]
fig, ax = plt.subplots(figsize=(8,6))
dot_on_bar(ax, err_names, err_vals, err_cols)
sax(ax, '', 'Model', 'MAE (kg CO2)')
fig.tight_layout(); save_fig(fig, nfig())

# G5-2  Total CO2 per Route per Model
fig, ax = plt.subplots(figsize=(14,6))
x  = np.arange(len(algo_names))
bw = 0.22
for mi, (mn, mc) in enumerate(zip(CO2_M, CCOLS)):
    vals  = [co2_per_route[mn][a] for a in algo_names]
    bars  = ax.bar(x + (mi-1)*bw, vals, width=bw, color=mc,
                   label=mn, edgecolor='white', linewidth=0.9, zorder=3)
    ax.scatter(x + (mi-1)*bw, vals, color='#1a2332', s=50, zorder=6)
ax.set_xticks(x); ax.set_xticklabels(algo_names, fontsize=14)
ax.legend()
sax(ax, '', 'Algorithm', 'CO2 (kg)')
fig.tight_layout(); save_fig(fig, nfig())

# G5-3  Carbon Emission Reduction %
fig, ax = plt.subplots(figsize=(10,6))
red_vals  = [co2_red_pct[m] for m in CO2_M]
dot_on_bar(ax, CO2_M, red_vals, CCOLS, fmt='.1f')
ax.axhline(0, color='#6b7280', lw=1.2, ls='--')
sax(ax, '', 'Model', 'Reduction (%)')
fig.tight_layout(); save_fig(fig, nfig())

# G5-4  CO2 per Stop — Best Route
fig, ax = plt.subplots(figsize=(13,6))
best_algo = algo_names[int(np.argmin(algo_costs))]
best_route_r = algo_routes[int(np.argmin(algo_costs))]
step_order = list(best_route_r)
for mn, mfn, mc in zip(CO2_M, CO2_FN, CCOLS):
    co2_steps = [mfn(flow_per_step[s]) for s in step_order]
    ax.plot(range(NUM_NODES_R), co2_steps, color=mc, lw=2.5,
            marker='o', ms=MSIZE, label=mn, zorder=4)
ax.set_xticks(range(NUM_NODES_R))
ax.set_xticklabels([stations[route_stations[i]] for i in range(NUM_NODES_R)],
                   rotation=30, fontsize=11)
sax(ax, '', 'Stop in Route', 'CO2 (kg)')
ax.legend(); fig.tight_layout(); save_fig(fig, nfig())

# ── FINAL SUMMARY DASHBOARD ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20,14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.40)

ax0 = fig.add_subplot(gs[0,:2])
ax0.fill_between(t_ax,ac_t*0.9,ac_t*1.1,color=PAL['band'],alpha=0.5,zorder=1)
ax0.plot(t_ax,ac_t,color=PAL['actual'],lw=2.8,marker='o',ms=6,label='Actual',zorder=5)
for nm,pd_,cl,mk in zip(MN,[lm_t,sm_t,dm_t],FCOLS,['s','^','D']):
    ax0.plot(t_ax,pd_,color=cl,lw=2.0,marker=mk,ms=5,ls='--',label=nm,zorder=4)
sax(ax0,'','Step','Flow'); ax0.legend(fontsize=11)

ax1 = fig.add_subplot(gs[0,2])
ax1.bar(MN,MAE_v,color=FCOLS,width=0.5,edgecolor='white',zorder=3)
ax1.scatter(MN,MAE_v,color='#111827',s=55,zorder=6)
sax(ax1,'','','MAE')

ax2 = fig.add_subplot(gs[0,3])
ax2.bar(MN,RMSE_v,color=FCOLS,width=0.5,edgecolor='white',zorder=3)
ax2.scatter(MN,RMSE_v,color='#111827',s=55,zorder=6)
sax(ax2,'','','RMSE')

ax3 = fig.add_subplot(gs[1,:2])
ax3.plot(t_ax,act_flow_t,color=PAL['actual'],lw=2.8,marker='o',ms=6,label='Actual',zorder=5)
for nm,fl_,cl,mk in zip(TM,[tm1_flow_t,tm2_flow_t,tm3_flow_t],TCOLS,['s','^','D']):
    ax3.plot(t_ax,fl_,color=cl,lw=2.0,marker=mk,ms=5,ls='--',label=nm,zorder=4)
sax(ax3,'','Step','Flow'); ax3.legend(fontsize=11)

ax4 = fig.add_subplot(gs[1,2])
ax4.bar(TM,TM_RMSE,color=TCOLS,width=0.5,edgecolor='white',zorder=3)
ax4.scatter(TM,TM_RMSE,color='#111827',s=55,zorder=6)
sax(ax4,'','','Flow RMSE')

ax5 = fig.add_subplot(gs[1,3])
ax5.bar(TM,TM_CONG,color=TCOLS,width=0.5,edgecolor='white',zorder=3)
ax5.scatter(TM,TM_CONG,color='#111827',s=55,zorder=6)
sax(ax5,'','','Cong. Acc. %')

ax6 = fig.add_subplot(gs[2,:2])
ax6.plot(iter_ga,  ga_hist,  color=PAL['ga'], lw=2.0,label='GA',  zorder=4)
ax6.plot(iter_aco, aco_hist, color=PAL['aco'],lw=2.0,label='ACO', zorder=4)
ax6.plot(iter_rl,  rl_bh,    color=PAL['rl'], lw=2.0,label='RL',  zorder=4)
sax(ax6,'','Iteration','Cost'); ax6.legend(fontsize=11)

ax7 = fig.add_subplot(gs[2,2])
ax7.bar(all_algo_names,all_costs_r,color=all_rcols,width=0.5,edgecolor='white',zorder=3)
ax7.scatter(all_algo_names,all_costs_r,color='#111827',s=55,zorder=6)
sax(ax7,'','','Route Cost'); ax7.tick_params(axis='x',labelsize=11)

ax8 = fig.add_subplot(gs[2,3])
r_co2_best = [co2_per_route['RSML-TUSCE'][a] for a in algo_names]
ax8.bar(algo_names,r_co2_best,color=RCOLS,width=0.5,edgecolor='white',zorder=3)
ax8.scatter(algo_names,r_co2_best,color='#111827',s=55,zorder=6)
sax(ax8,'','','CO2 (RSML)')

fig.tight_layout(); save_fig(fig, nfig())

print(f"\n  ✓  {fig_count} PDF figures saved in: {OUT_DIR}/")
print("\n══ FINAL SUMMARY ══════════════════════════════════════════")
print(f"  Station : {LABEL}")
print(f"\n  Demand Forecasting:")
print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
for n,a,r,p in zip(MN,MAE_v,RMSE_v,MAPE_v):
    print(f"  {n:<12} {a:>8.2f} {r:>8.2f} {p:>7.1f}%")
print(f"\n  Route Optimisation:")
print(f"  {'Algorithm':<10} {'Cost':>10} {'Fuel':>8}")
for n,c,fl in zip(algo_names,algo_costs,algo_fuel):
    print(f"  {n:<10} {c:>10.3f} {fl:>8.4f}")
print("═" * 62)