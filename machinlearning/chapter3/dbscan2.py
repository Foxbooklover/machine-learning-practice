from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from scipy.cluster.hierarchy import dendrogram, ward


X, y = make_blobs(random_state=0, n_samples=12)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ===== 데이터 준비 (예: 사용자가 이미 만든 X 사용) =====
# X, y = make_blobs(random_state=0, n_samples=60, centers=3, cluster_std=0.8)
# 여기서는 사용자가 만든 X를 그대로 쓴다고 가정
# 예시용:
# from sklearn.datasets import make_blobs
# X, _ = make_blobs(random_state=0, n_samples=60, centers=3, cluster_std=0.8)

# ------- 파라미터 설정 -------
eps = 1.5
min_samples = 3

# ===== 유틸 함수들 =====
def pairwise_radius_neighbors(X, eps):
    """각 포인트별로 반경 eps 내부 이웃들의 인덱스 리스트를 반환"""
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    neighbors = [np.where(D[i] <= eps)[0] for i in range(len(X))]
    return neighbors

def core_border_noise(neighbors, min_samples):
    """core/border/noise 마스크 반환"""
    n = len(neighbors)
    core_mask = np.array([len(neighbors[i]) >= min_samples for i in range(n)])
    # border: core는 아니지만, core의 이웃에 속하는 점
    border_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not core_mask[i]:
            # 이 점 i의 이웃 중 core가 하나라도 있으면 border
            if np.any(core_mask[neighbors[i]]):
                border_mask[i] = True
    noise_mask = ~(core_mask | border_mask)
    return core_mask, border_mask, noise_mask

def connected_components_among_core(neighbors, core_mask):
    """core 포인트들만으로 그래프를 만들고 연결요소(컴포넌트) 리스트를 반환"""
    n = len(neighbors)
    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if core_mask[i] and not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                # core-core 간에만 연결
                for v in neighbors[u]:
                    if core_mask[v] and not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(np.array(comp, dtype=int))
    return components

def assign_border_to_components(neighbors, core_mask, components, n_points):
    """border 포인트를 인접 core의 컴포넌트 라벨로 할당 (여러 개면 임의로 첫번째에 할당)"""
    # 컴포넌트 id 맵 만들기
    comp_id = -np.ones(n_points, dtype=int)
    for k, comp in enumerate(components):
        comp_id[comp] = k

    labels_border = -np.ones(n_points, dtype=int)
    for i in range(n_points):
        if core_mask[i]:
            labels_border[i] = comp_id[i]
        else:
            # 이웃 중 core가 있으면 그 core의 comp id를 하나 가져옴
            cands = [comp_id[j] for j in neighbors[i] if core_mask[j]]
            if len(cands) > 0:
                labels_border[i] = cands[0]  # 여러 개면 첫번째
    return labels_border  # core와 border는 0..K-1, noise는 -1

def get_bounds(X, pad_ratio=0.05):
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    return (xmin - dx*pad_ratio, xmax + dx*pad_ratio,
            ymin - dy*pad_ratio, ymax + dy*pad_ratio)

# ===== 단계별 스냅샷 구성 =====
def build_dbscan_snapshots(X, eps=1.5, min_samples=3):
    snapshots = []
    n = len(X)

    # 공통 값들 계산
    neighbors = pairwise_radius_neighbors(X, eps)
    core_mask, border_mask, noise_mask = core_border_noise(neighbors, min_samples)
    components = connected_components_among_core(neighbors, core_mask)
    labels_partial = -np.ones(n, dtype=int)  # 점진적 라벨링용 (확장 단계에서 사용)
    final_labels = assign_border_to_components(neighbors, core_mask, components, n)

    # (1) Raw data
    snapshots.append(dict(
        title="Step 1: Raw data",
        mode="raw",
        data=dict()
    ))

    # (2) ε 이웃/연결관계(전체) — 모든 점 쌍 중 거리<=eps인 엣지 표시
    edges_all = []
    for i in range(n):
        for j in neighbors[i]:
            if j > i:
                edges_all.append((i, j))
    snapshots.append(dict(
        title=f"Step 2: ε-neighborhood graph (eps={eps})",
        mode="edges_all",
        data=dict(edges=edges_all)
    ))

    # (3) Core / Border / Noise 판별
    snapshots.append(dict(
        title=f"Step 3: Core / Border / Noise (min_samples={min_samples})",
        mode="cbn",
        data=dict(core_mask=core_mask, border_mask=border_mask, noise_mask=noise_mask)
    ))

    # (4) Core 그래프(코어들 사이 엣지)
    edges_core = []
    for i in range(n):
        if not core_mask[i]:
            continue
        for j in neighbors[i]:
            if j > i and core_mask[j]:
                edges_core.append((i, j))
    snapshots.append(dict(
        title="Step 4: Core graph (connected core points)",
        mode="edges_core",
        data=dict(edges=edges_core, core_mask=core_mask)
    ))

    # (5~) 컴포넌트별(클러스터) 확장 — 코어만 먼저 단계적으로 색칠
    for idx, comp in enumerate(components, start=1):
        labels_partial = labels_partial.copy()
        labels_partial[comp] = idx-1  # 0부터 부여
        snapshots.append(dict(
            title=f"Step 5.{idx}: Expand cluster {idx} (core only)",
            mode="expand_core",
            data=dict(labels_partial=labels_partial, core_mask=core_mask)
        ))

    # (마지막-1) Border 부착(코어 클러스터에 인접한 보더 점들 색칠)
    snapshots.append(dict(
        title="Step 6: Attach border points to nearest core clusters",
        mode="attach_border",
        data=dict(core_mask=core_mask, labels_border=final_labels)
    ))

    # (마지막) 최종 결과 (sklearn DBSCAN 결과와 동일 로직)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    snapshots.append(dict(
        title="Step 7: Final labels (DBSCAN)",
        mode="final",
        data=dict(labels=db.labels_)
    ))

    return snapshots

# ===== 시각화 =====
def plot_snapshots(X, snapshots, point_size=60):
    xmin, xmax, ymin, ymax = get_bounds(X)
    n_steps = len(snapshots)

    # 자동 그리드 설정
    ncols = 4 if n_steps >= 4 else n_steps
    nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # 공통 팔레트
    cmap = plt.cm.get_cmap("tab10")

    for k, snap in enumerate(snapshots):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        mode = snap["mode"]
        ax.set_title(snap["title"])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xticks([]); ax.set_yticks([])

        if mode == "raw":
            ax.scatter(X[:, 0], X[:, 1], s=point_size, c="gray")

        elif mode == "edges_all":
            # 모든 ε-이웃 엣지
            for (i, j) in snap["data"]["edges"]:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], alpha=0.3, linewidth=1)
            ax.scatter(X[:, 0], X[:, 1], s=point_size, c="gray")

        elif mode == "cbn":
            core_mask = snap["data"]["core_mask"]
            border_mask = snap["data"]["border_mask"]
            noise_mask = snap["data"]["noise_mask"]
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1], s=point_size, c="red", label="noise", alpha=0.9)
            ax.scatter(X[border_mask, 0], X[border_mask, 1], s=point_size, c="orange", label="border", alpha=0.9)
            ax.scatter(X[core_mask, 0], X[core_mask, 1], s=point_size, c="blue", label="core", alpha=0.9)
            ax.legend(loc="best", fontsize=9)

        elif mode == "edges_core":
            edges = snap["data"]["edges"]
            core_mask = snap["data"]["core_mask"]
            # 엣지 먼저
            for (i, j) in edges:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], alpha=0.5, linewidth=1.5)
            # 코어/비코어 표시
            ax.scatter(X[~core_mask, 0], X[~core_mask, 1], s=point_size, c="lightgray")
            ax.scatter(X[core_mask, 0], X[core_mask, 1], s=point_size, c="blue")

        elif mode == "expand_core":
            labels_partial = snap["data"]["labels_partial"]
            core_mask = snap["data"]["core_mask"]
            # 아직 미할당 core는 회색, 할당된 core는 라벨 색
            unassigned_core = core_mask & (labels_partial < 0)
            assigned_core = core_mask & (labels_partial >= 0)
            ax.scatter(X[~core_mask, 0], X[~core_mask, 1], s=point_size, c="lightgray")
            if np.any(unassigned_core):
                ax.scatter(X[unassigned_core, 0], X[unassigned_core, 1], s=point_size, c="lightgray")
            if np.any(assigned_core):
                ax.scatter(X[assigned_core, 0], X[assigned_core, 1], s=point_size,
                           c=labels_partial[assigned_core], cmap=cmap, vmin=0)

        elif mode == "attach_border":
            core_mask = snap["data"]["core_mask"]
            labels_border = snap["data"]["labels_border"]
            # noise
            noise = labels_border < 0
            ax.scatter(X[noise, 0], X[noise, 1], s=point_size, c="red", alpha=0.9, label="noise")
            # core+border (라벨>=0)
            assigned = labels_border >= 0
            ax.scatter(X[assigned, 0], X[assigned, 1], s=point_size,
                       c=labels_border[assigned], cmap=cmap, vmin=0)
            ax.legend(loc="best", fontsize=9)

        elif mode == "final":
            labels = snap["data"]["labels"]
            noise = labels < 0
            ax.scatter(X[noise, 0], X[noise, 1], s=point_size, c="red", label="noise")
            assigned = labels >= 0
            ax.scatter(X[assigned, 0], X[assigned, 1], s=point_size,
                       c=labels[assigned], cmap=cmap, vmin=0)
            ax.legend(loc="best", fontsize=9)

    # 빈 subplot 처리
    total_axes = nrows * ncols
    for k in range(n_steps, total_axes):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

# ===== 실행 =====
snapshots = build_dbscan_snapshots(X, eps=1.5, min_samples=3)
plot_snapshots(X, snapshots)
