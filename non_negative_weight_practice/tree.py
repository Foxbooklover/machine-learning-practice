import matplotlib.pyplot as plt
import mglearn
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

nodes = {
    "Machine Learning": (0, 0),

    "Supervised": (1, 1),
    "Unsupervised": (1, -1),
    "Reinforcement": (1, -3),

    "regression": (2, 1.5),
    "classification": (2, 0.5),
    "clustering": (2, -0.5),
    "dimensionality reduction": (2, -1.5),
    "Value based": (2, -2.5),
    "Policy based": (2, -3.5),

    "logistic regression": (3, 3),
    "Decision Tree": (3, 2),
    "SVM": (3, 1.5),
    "MLP": (3, 1),
    "KMeans": (3, 0),
    "DBSCAN": (3, -0.5),           # 대문자 표기 권장
    "agglomerative clustering": (3, -1),
    "PCA": (3, -2),
    "Q-Learning": (3, -3),
    "Policy Gradient": (3, -4),

    "random forest": (4, 2),
    "gradient boosting": (4, 1),
}

edges = [
    ("Machine Learning", "Supervised"),
    ("Machine Learning", "Unsupervised"),
    ("Machine Learning", "Reinforcement"),

    ("Supervised", "regression"),
    ("Supervised", "classification"),
    ("Unsupervised", "clustering"),
    ("Unsupervised", "dimensionality reduction"),
    ("Reinforcement", "Value based"),
    ("Reinforcement", "Policy based"),

    # (참고) 보통 logistic regression은 classification에 속함
    ("classification", "logistic regression"),
    ("classification", "Decision Tree"),
    ("classification", "SVM"),
    ("classification", "MLP"),
    ("clustering", "KMeans"),
    ("clustering", "DBSCAN"),
    ("clustering", "agglomerative clustering"),
    ("dimensionality reduction", "PCA"),
    ("Value based", "Q-Learning"),
    ("Policy based", "Policy Gradient"),
    ("Decision Tree", "random forest"),
    ("Decision Tree", "gradient boosting"),
]
plt.figure(figsize=(10, 6))

# 간선 그리기
for start, end in edges:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    plt.plot([x1, x2], [y1, y2], "k-", zorder=1)

# 노드 이름만 표시 (원 없음)
for node, (x, y) in nodes.items():
    plt.text(x, y, node, ha="center", va="center", fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"))


plt.axis("off")
plt.show()


'''
# ---- 배열로 변환 & 레벨(열) 색상 매핑 ----
names = list(nodes.keys())
XY = np.array([nodes[n] for n in names])          # (N,2)
X = XY[:, 0]
Y = XY[:, 1]

unique_x = np.unique(X)
x_to_level = {x: i for i, x in enumerate(unique_x)}   # x값 -> 0..K-1
levels = np.array([x_to_level[x] for x in X])

# colormap을 이산 구간으로 사용
cmap = plt.cm.tab20c
bounds = np.arange(len(unique_x)+1) - 0.5            # [-0.5, 0.5, 1.5, ...]
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(11, 6))

# 간선 먼저(뒤에 점이 위에 보이도록)
for s, e in edges:
    x1, y1 = nodes[s]
    x2, y2 = nodes[e]
    plt.plot([x1, x2], [y1, y2], "k-", linewidth=1, zorder=1)

# 노드 한 번에 그리기(레벨별 색)
plt.scatter(X, Y, c=levels, cmap=cmap, norm=norm,
            s=1200, edgecolors="black", zorder=2)

# 라벨
for name, (x, y) in nodes.items():
    plt.text(x, y, name, ha="center", va="center", fontsize=9, zorder=3)

# 레벨(열) 범례 만들기
legend_handles = []
for x in unique_x:
    level = x_to_level[x]
    color = cmap(norm(level))
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
               markeredgecolor='black', markersize=10, label=f"x = {x}")
    )
plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1))

plt.axis("off")
plt.tight_layout()
plt.show()
'''