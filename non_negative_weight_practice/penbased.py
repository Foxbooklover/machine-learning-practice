# pendigits_mlp_savecsv.py
# PenDigits 분류 MLP (은닉 [9,7]) + 모든 가중치/바이어스 CSV 저장
# 테스트 전체 샘플에 대한 logits(점수), softmax 확률, 예측/정답 CSV 저장

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ===================== 설정 =====================
NONNEG = True           # 비음수 가중치 강제 여부 (softplus 재파라미터화)
SEED = 42
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
HIDDEN = (9, 7)         # 은닉층 크기
EXPORT_DIR = "exports_pendigits"
os.makedirs(EXPORT_DIR, exist_ok=True)
# ===================== 시드 고정 =====================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device} | Non-Negative Weights: {NONNEG} | Hidden={HIDDEN}")

# ===================== 데이터 로드 =====================
# OpenML 'pendigits' (특징 16, 클래스 10)
Xy = fetch_openml('pendigits', version=1, as_frame=False)
X = Xy.data.astype(np.float32)              # (N, 16)
y = Xy.target.astype(int)                   # (N,)

# train/val/test = 0.8/0.1/0.1
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
)

# 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Tensor
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_val_t   = torch.from_numpy(X_val)
y_val_t   = torch.from_numpy(y_val.astype(np.int64))
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test.astype(np.int64))

# DataLoader
train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
val_ds   = torch.utils.data.TensorDataset(X_val_t, y_val_t)
test_ds  = torch.utils.data.TensorDataset(X_test_t, y_test_t)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===================== Non-Negative Linear =====================
class NonNegativeLinear(nn.Module):
    """
    y = x @ softplus(V).T + b   => 모든 weight >= 0 보장
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.V, a=np.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / np.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        W = F.softplus(self.V)  # 항상 양수
        y = x @ W.T
        if self.bias is not None:
            y = y + self.bias
        return y

def LinearFactory(in_f, out_f, bias=False, nonneg=False):
    return NonNegativeLinear(in_f, out_f, bias) if nonneg else nn.Linear(in_f, out_f, bias)

# ===================== 모델 정의 =====================
class MLP(nn.Module):
    def __init__(self, in_dim=16, hidden=(9,7), out_dim=10, nonneg=False):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden)
        for i in range(len(dims)-1):
            layers.append(LinearFactory(dims[i], dims[i+1], bias=False, nonneg=nonneg))
            layers.append(nn.ReLU())
        layers.append(LinearFactory(dims[-1], out_dim, bias=False, nonneg=nonneg))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=X_train.shape[1], hidden=HIDDEN, out_dim=10, nonneg=NONNEG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# ===================== 평가 함수 =====================
def evaluate(loader):
    model.eval()
    loss_sum, n = 0.0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = model(xb)
            loss = criterion(z, yb)
            loss_sum += loss.item() * yb.size(0)
            n += yb.size(0)
            all_pred.append(torch.argmax(z, dim=1).cpu().numpy())
            all_true.append(yb.cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    acc = accuracy_score(all_true, all_pred)
    return loss_sum / n, acc

# ===================== 학습 =====================
best_val_acc, best_state = 0.0, None
for epoch in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        z = model(xb)
        loss = criterion(z, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    tr_loss, tr_acc = evaluate(train_loader)
    va_loss, va_acc = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | Train acc {tr_acc:.4f} | Val acc {va_acc:.4f}")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

# ===================== 테스트 & 리포트 =====================
test_loss, test_acc = evaluate(test_loader)
print(f"[Test] acc={test_acc:.4f}")

model.eval()
with torch.no_grad():
    z_test = model(X_test_t.to(device))                 # (N_test, 10) logits
    probs_test = F.softmax(z_test, dim=1)               # (N_test, 10)
    preds_test = torch.argmax(z_test, dim=1)            # (N_test,)

z_test_np = z_test.cpu().numpy().astype(np.float32)
probs_test_np = probs_test.cpu().numpy().astype(np.float32)
preds_test_np = preds_test.cpu().numpy().astype(np.int32)
y_test_np = y_test.astype(np.int32)

print(classification_report(y_test_np, preds_test_np, digits=4))

# ===================== CSV 저장 유틸 =====================
def save_csv(array: np.ndarray, path: str, header: str = None):
    if header is not None:
        np.savetxt(path, array, delimiter=",", fmt="%.7g", header=header, comments="")
    else:
        np.savetxt(path, array, delimiter=",", fmt="%.7g")

# ----- 3) 외부 테스트용 데이터 저장 (전처리 완료 버전) -----
# X_test: 표준화 완료된 float32, shape (N_test, 16)
# y_test: 정답 라벨 int, shape (N_test,)

save_csv(X_test, os.path.join(EXPORT_DIR, "X_test.csv"),
         header="feat" + ",".join(str(i) for i in range(X_test.shape[1])))

save_csv(y_test.reshape(-1, 1), os.path.join(EXPORT_DIR, "y_test.csv"),
         header="label")


# ----- 1) 모든 레이어 가중치/바이어스 저장 -----
layer_idx = 0
for m in model.net:
    if isinstance(m, (nn.Linear, NonNegativeLinear)):
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().astype(np.float32)   # (out, in)
        else:
            # NonNegativeLinear: 실제 weight = softplus(V)
            W = F.softplus(m.V).detach().cpu().numpy().astype(np.float32)
        save_csv(W, os.path.join(EXPORT_DIR, f"layer{layer_idx}_weight.csv"))
        if m.bias is not None:
            b = m.bias.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
            save_csv(b, os.path.join(EXPORT_DIR, f"layer{layer_idx}_bias.csv"))
        layer_idx += 1

# ----- 2) 테스트 전체 점수/확률/예측 저장 -----
save_csv(z_test_np,      os.path.join(EXPORT_DIR, "test_logits.csv"),
         header="z_class0,z_class1,z_class2,z_class3,z_class4,z_class5,z_class6,z_class7,z_class8,z_class9")
save_csv(probs_test_np,  os.path.join(EXPORT_DIR, "test_probs.csv"),
         header="p_class0,p_class1,p_class2,p_class3,p_class4,p_class5,p_class6,p_class7,p_class8,p_class9")

# preds + y_true + correct
pred_pack = np.column_stack([y_test_np, preds_test_np, (y_test_np == preds_test_np).astype(np.int32)])
save_csv(pred_pack, os.path.join(EXPORT_DIR, "test_predictions.csv"),
         header="y_true,y_pred,correct")

print(f"[Saved] Weights/Biases & Test scores exported to: ./{EXPORT_DIR}/")
