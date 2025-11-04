import numpy as np
from pathlib import Path

def load_csv(name, skiprows=0):  # 기본값은 0
    return np.loadtxt(name, delimiter=",", skiprows=skiprows)


def relu(x):  # ReLU 활성화
    return np.maximum(x, 0.0)

# 1) 데이터 로드
# header 한 줄 있는 파일들
X   = load_csv("X_test.csv", skiprows=1)
y   = load_csv("y_test.csv", skiprows=1)
LOG_ref = load_csv("test_logits.csv", skiprows=1)

W0  = load_csv("layer0_weight.csv", skiprows=0)
W1  = load_csv("layer1_weight.csv", skiprows=0)
W2  = load_csv("layer2_weight.csv", skiprows=0)



print("Shapes -> X", X.shape, "W0", W0.shape, "W1", W1.shape, "W2", W2.shape, "LOG_ref", LOG_ref.shape)

def try_all_with_relu(X, W0, W1, W2, LOG_ref):
    Ws0, Ws1, Ws2 = [W0, W0.T], [W1, W1.T], [W2, W2.T]
    best = None
    for A in Ws0:
        for B in Ws1:
            for C in Ws2:
                try:
                    H1 = relu(X  @ A.T)
                    H2 = relu(H1 @ B.T)
                    LOG = H2 @ C.T
                except Exception:
                    continue
                if LOG.shape != LOG_ref.shape:
                    continue
                diff = np.max(np.abs(LOG - LOG_ref))
                mae  = np.mean(np.abs(LOG - LOG_ref))
                rmse = np.sqrt(np.mean((LOG - LOG_ref)**2))
                score = diff  # 우선 max-오차를 최소화
                if (best is None) or (score < best[0]):
                    best = (score, mae, rmse, A is W0, B is W1, C is W2, LOG)
    return best

best = try_all_with_relu(X, W0, W1, W2, LOG_ref)

if best is None:
    print("\n형상 불일치로 곱셈을 구성하지 못했습니다.")
else:
    diff, mae, rmse, a_orig, b_orig, c_orig, LOG = best
    print(f"\n최소 오차 조합(활성화=ReLU):")
    print(f"  max|Δ|={diff:.8g},  MAE={mae:.8g},  RMSE={rmse:.8g}")
    print("  W0 사용형태:", "원본" if a_orig else "전치(T)")
    print("  W1 사용형태:", "원본" if b_orig else "전치(T)")
    print("  W2 사용형태:", "원본" if c_orig else "전치(T)")

    tol = 1e-6
    print(f"\n로그릿 일치(@ tol={tol}):", "완전 일치 ✅" if diff < tol else "약간 차이 ⚠️")

    # (선택) 예측값도 있으면 비교
    try:
        y_pred_ref = load_csv("test_predictions").astype(int)
        y_pred_ours = np.argmax(LOG, axis=1)
        acc = (y_pred_ours == y_pred_ref).mean()
        print(f"\n예측 레이블 일치율: {acc*100:.2f}%")
    except Exception:
        pass

    # 샘플 일부 비교
    for i in range(min(3, LOG.shape[0])):
        print(f"\n[i={i}] ours[:5]={np.array2string(LOG[i,:5], precision=6)}")
        print(f"[i={i}] ref [:5]={np.array2string(LOG_ref[i,:5], precision=6)}")
