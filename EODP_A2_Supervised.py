import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_validate, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
import numpy as np

from EODP_A2_Preprocessing import df, df_rating

RANDOM_STATE = 42

# Target Construction

print("\n" + "="*60)
print("2.3 SUPERVISED LEARNING - target variable construction")
print("="*60)

# Rated subset only: review_scores_rating is a core feature for the RQ, and its
# missingness is MNAR (every null = 0-review listing), so imputing would invent ratings.
base = df_rating.dropna(subset=['price']).copy()

print(f"all listings: {len(df)}")
print(f"rated subset: {len(df_rating)}")
print(f"rated + priced: {len(base)}  (dropped {len(df_rating) - len(base)} rows with NaN price)")

print("\nprice extremes:")
print(f"  min {base['price'].min():.2f} | median {base['price'].median():.2f} | max {base['price'].max():.2f}")
print(f"  price == 0: {(base['price'] == 0).sum()} rows")
print(f"  price > 1000: {(base['price'] > 1000).sum()} rows")

# Tertile cut points, then applied so that a listing priced exactly on a cut point
# falls in the upper band: low is below the first edge, mid spans both edges
# inclusive, high is above the second. pd.qcut is right-closed and would instead put
# the 2 listings at exactly $194.25 in low.
_, bins = pd.qcut(base['price'], q=3, retbins=True)
base['price_band'] = pd.cut(base['price'],
                            bins=[-np.inf, bins[1] - 1e-9, bins[2], np.inf],
                            labels=['low', 'mid', 'high'], right=True)

print("\ncut points ($):", [round(b, 2) for b in bins])
print(f"  low:  {bins[0]:.2f} - {bins[1]:.2f}")
print(f"  mid:  {bins[1]:.2f} - {bins[2]:.2f}")
print(f"  high: {bins[2]:.2f} - {bins[3]:.2f}")

counts = base['price_band'].value_counts().sort_index()
print("\nclass balance:")
for band, n in counts.items():
    print(f"  {band:5s} {n:6d}  ({n/len(base)*100:.1f}%)")

majority_baseline = counts.max() / len(base)
print(f"\nmajority-class (0R) baseline accuracy: {majority_baseline:.4f}")


print("\n" + "="*60)


# Encoding the Column

print("FEATURE MATRIX - encoding")
print("="*60)

# bedrooms is the only chosen feature with meaningful missingness, so check whether
# the nulls track room_type before picking a fill strategy.
print("\nbedrooms nulls by room_type:")
for rt, grp in base.groupby('room_type'):
    n_null = grp['bedrooms'].isna().sum()
    print(f"  {rt:16s} {n_null:5d} / {len(grp):5d}  ({n_null/len(grp)*100:5.1f}%)")

base['bedrooms'] = base.groupby('room_type')['bedrooms'].transform(lambda s: s.fillna(s.median()))
base['bedrooms'] = base['bedrooms'].fillna(base['bedrooms'].median())
print(f"\nbedrooms nulls after within-room_type median impute: {base['bedrooms'].isna().sum()}")

NUMERIC = ['review_scores_rating', 'accommodates', 'bedrooms', 'distance_from_cbd_km']
CATEGORICAL = ['room_type', 'property_group']

# drop_first=False keeps every level as its own column so the tree's feature
# importances stay readable for the feature-selection section.
X = pd.get_dummies(base[NUMERIC + CATEGORICAL], columns=CATEGORICAL, drop_first=False)
X = X.astype(float)
y = base['price_band']

print(f"\nnumeric features      : {len(NUMERIC)}  {NUMERIC}")
print(f"categorical features  : {len(CATEGORICAL)} -> one-hot expanded")
print(f"\nX shape: {X.shape}   y shape: {y.shape}")
print(f"total feature columns after encoding: {X.shape[1]}")
print("\ncolumns:")
for i, c in enumerate(X.columns, 1):
    print(f"  {i:2d}. {c}")
print(f"\nany nulls left in X: {X.isna().sum().sum()}")


print("\n" + "="*60)
print("TRAIN / TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print(f"\ntrain: {len(X_train)} rows | test: {len(X_test)} rows  (80/20)")
print("\nstratification check (share per band):")
print(f"  {'band':6s} {'full':>8s} {'train':>8s} {'test':>8s}")
for band in ['low', 'mid', 'high']:
    print(f"  {band:6s} {(y == band).mean():8.4f} {(y_train == band).mean():8.4f} {(y_test == band).mean():8.4f}")

# KNN compares listings by distance, so features on wider ranges would dominate.
# Fitted on train only: fitting on all rows would leak test distribution into training.
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nscaled copies built for KNN (scaler fitted on train only)")


print("\n" + "="*60)
print("BASELINE + DEFAULT MODELS")
print("="*60)

def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    print(f"  {name:34s} accuracy {acc:.4f} | macro-F1 {macro:.4f}")
    return acc, macro

dummy = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)

knn_default = KNeighborsClassifier()
knn_default.fit(X_train_scaled, y_train)

tree_default = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_default.fit(X_train, y_train)

print("\ntest-set performance:")
base_acc, base_f1 = report("majority baseline (0R)", y_test, dummy.predict(X_test))
knn_acc, knn_f1 = report("KNN (default k=5)", y_test, knn_default.predict(X_test_scaled))
tree_acc, tree_f1 = report("Decision Tree (default, unpruned)", y_test, tree_default.predict(X_test))

print("\nimprovement over baseline (absolute, accuracy):")
print(f"  KNN           {knn_acc - base_acc:+.4f}")
print(f"  Decision Tree {tree_acc - base_acc:+.4f}")

print(f"\ntree depth when unconstrained: {tree_default.get_depth()} | leaves: {tree_default.get_n_leaves()}")
print(f"tree train accuracy: {accuracy_score(y_train, tree_default.predict(X_train)):.4f}"
      f"  (vs test {tree_acc:.4f}) -> overfitting gap")


print("\n" + "="*60)
print("FEATURE SET CHOICE: keep or drop 'bedrooms'")
print("="*60)
print("\nSpearman accommodates~bedrooms = 0.8643 (2.2, n = 14,058), so the two")
print("carry largely the same information. Decided by 5-fold CV on the TRAINING")
print("set only, leaving the test set untouched for the final reported metric.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Scaler lives inside the pipeline so it refits on each CV fold's training
# portion rather than seeing the whole training set.
knn_pipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])

feature_sets = {
    'keep both': list(X_train.columns),
    'drop bedrooms': [c for c in X_train.columns if c != 'bedrooms'],
    'drop accommodates': [c for c in X_train.columns if c != 'accommodates'],
}

print(f"\n{'feature set':20s} {'cols':>5s} {'KNN macro-F1':>16s} {'Tree macro-F1':>16s}")
cv_results = {}
for label, cols in feature_sets.items():
    knn_cv = cross_val_score(knn_pipe, X_train[cols], y_train,
                             cv=cv, scoring='f1_macro')
    tree_cv = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE),
                              X_train[cols], y_train, cv=cv, scoring='f1_macro')
    cv_results[label] = (knn_cv, tree_cv)
    print(f"{label:20s} {len(cols):5d} {knn_cv.mean():10.4f} +/-{knn_cv.std():.4f} "
          f"{tree_cv.mean():10.4f} +/-{tree_cv.std():.4f}")

print("\nchange in CV macro-F1 relative to keeping both:")
for label in ['drop bedrooms', 'drop accommodates']:
    dk = cv_results[label][0].mean() - cv_results['keep both'][0].mean()
    dt = cv_results[label][1].mean() - cv_results['keep both'][1].mean()
    print(f"  {label:20s} KNN {dk:+.4f} | Tree {dt:+.4f}")


print("\n" + "="*72)
print("HYPERPARAMETER SWEEP - 5-fold stratified CV on training set only")
print("="*72)

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41]
DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, None]

print("\n--- KNN: n_neighbors x weights (sklearn defaults = 5, 'uniform') ---")
print("uniform  = all k neighbours vote equally")
print("distance = each neighbour's vote weighted by 1/distance\n")
print(f"{'k':>4s} {'uniform F1':>12s} {'std':>8s} {'distance F1':>13s} {'std':>8s}")
knn_sweep = []
for k in K_VALUES:
    row = {}
    for w in ['uniform', 'distance']:
        pipe = Pipeline([('scale', StandardScaler()),
                         ('knn', KNeighborsClassifier(n_neighbors=k, weights=w))])
        res = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring='f1_macro', return_train_score=True)
        row[w] = (res['test_score'].mean(), res['test_score'].std(),
                  res['train_score'].mean())
        knn_sweep.append((k, w, res['test_score'].mean(), res['test_score'].std(),
                          res['train_score'].mean()))
    print(f"{k:4d} {row['uniform'][0]:12.4f} {row['uniform'][1]:8.4f} "
          f"{row['distance'][0]:13.4f} {row['distance'][1]:8.4f}")

best_knn = max(knn_sweep, key=lambda r: r[2])
best_k = (best_knn[0], best_knn[2])
print(f"\nbest KNN: k={best_knn[0]}, weights='{best_knn[1]}'  "
      f"(CV macro-F1 {best_knn[2]:.4f} +/-{best_knn[3]:.4f})")

best_uniform = max((r for r in knn_sweep if r[1] == 'uniform'), key=lambda r: r[2])
best_distance = max((r for r in knn_sweep if r[1] == 'distance'), key=lambda r: r[2])
print(f"  best uniform : k={best_uniform[0]:3d}  {best_uniform[2]:.4f}")
print(f"  best distance: k={best_distance[0]:3d}  {best_distance[2]:.4f}")
print(f"  effect of weights='distance': {best_distance[2] - best_uniform[2]:+.4f}")

print("\n--- Decision Tree: max_depth (sklearn default = None, unlimited) ---")
print(f"{'depth':>6s} {'CV macro-F1':>13s} {'std':>8s} {'train F1':>10s} {'gap':>8s}")
tree_sweep = []
for d in DEPTH_VALUES:
    res = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE),
                         X_train, y_train, cv=cv,
                         scoring='f1_macro', return_train_score=True)
    gap = res['train_score'].mean() - res['test_score'].mean()
    tree_sweep.append((d, res['test_score'].mean(), res['test_score'].std(),
                       res['train_score'].mean(), gap))
    label = 'None' if d is None else str(d)
    print(f"{label:>6s} {res['test_score'].mean():13.4f} {res['test_score'].std():8.4f} "
          f"{res['train_score'].mean():10.4f} {gap:8.4f}")

best_d = max(tree_sweep, key=lambda r: r[1])
print(f"\nbest max_depth = {best_d[0]}  (CV macro-F1 {best_d[1]:.4f})")

print("\n--- hyperparameter justification (spec 3.4: default / chosen / effect) ---")
default_knn = [r for r in knn_sweep if r[0] == 5 and r[1] == 'uniform'][0]
default_tree = [r for r in tree_sweep if r[0] is None][0]
print(f"KNN n_neighbors : default 5 -> chosen {best_k[0]}  "
      f"effect on CV macro-F1 {best_k[1] - default_knn[2]:+.4f}")
print(f"Tree max_depth  : default None -> chosen {best_d[0]}  "
      f"effect on CV macro-F1 {best_d[1] - default_tree[1]:+.4f}")
print(f"Tree overfitting gap shrinks from {default_tree[4]:.4f} (None) "
      f"to {best_d[4]:.4f} (depth {best_d[0]})")


print("\n" + "="*72)
print("FINAL MODELS - refit at chosen hyperparameters, scored on held-out test set")
print("="*72)

# k=25 over the tied k=31: same CV mean, lower fold variance, simpler model.
CHOSEN_K = 25
CHOSEN_DEPTH = 8

knn_final = Pipeline([('scale', StandardScaler()),
                      ('knn', KNeighborsClassifier(n_neighbors=CHOSEN_K))]).fit(X_train, y_train)
tree_final = DecisionTreeClassifier(max_depth=CHOSEN_DEPTH,
                                    random_state=RANDOM_STATE).fit(X_train, y_train)

LABELS = ['low', 'mid', 'high']
models = {
    f'KNN (k={CHOSEN_K})': knn_final.predict(X_test),
    f'Decision Tree (depth={CHOSEN_DEPTH})': tree_final.predict(X_test),
    'Majority baseline (0R)': dummy.predict(X_test),
}

print("\n--- headline test-set metrics ---")
print(f"{'model':32s} {'accuracy':>10s} {'macro-F1':>10s} {'gain vs 0R':>12s}")
baseline_acc = accuracy_score(y_test, models['Majority baseline (0R)'])
final_scores = {}
for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average='macro')
    final_scores[name] = (acc, mf1)
    gain = 0.0 if 'baseline' in name else acc - baseline_acc
    print(f"{name:32s} {acc:10.4f} {mf1:10.4f} {gain:+12.4f}")

print("\n--- per-class precision / recall / F1 ---")
for name, preds in models.items():
    if 'baseline' in name:
        continue
    print(f"\n{name}:")
    print(classification_report(y_test, preds, labels=LABELS, digits=4, zero_division=0))

print("--- confusion matrices (rows = actual, cols = predicted) ---")
for name, preds in models.items():
    if 'baseline' in name:
        continue
    cm = confusion_matrix(y_test, preds, labels=LABELS)
    print(f"\n{name}")
    print(f"{'':>8s}" + "".join(f"{l:>8s}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:>8s}" + "".join(f"{v:8d}" for v in cm[i]))


print("\n" + "="*72)
print("UNCERTAINTY - bootstrap 95% CI on test macro-F1")
print("="*72)
print("\nResampling the test set with replacement 2000 times. This quantifies how much")
print("the reported score depends on which listings happened to land in the test split.")

N_BOOT = 2000
rng = np.random.default_rng(RANDOM_STATE)
y_test_arr = np.asarray(y_test)

for name, preds in models.items():
    preds_arr = np.asarray(preds)
    scores = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y_test_arr), len(y_test_arr))
        scores[b] = f1_score(y_test_arr[idx], preds_arr[idx],
                             average='macro', zero_division=0)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"  {name:32s} macro-F1 {scores.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


print("\n" + "="*72)
print("FEATURE INFLUENCE - Decision Tree embedded importances (for Ben, 3.5)")
print("="*72)
importances = (pd.Series(tree_final.feature_importances_, index=X_train.columns)
               .sort_values(ascending=False))
print()
for rank, (feat, imp) in enumerate(importances.items(), 1):
    marker = '  <-- top 3' if rank <= 3 else ''
    print(f"  {rank:2d}. {feat:32s} {imp:.4f}{marker}")


print("\n" + "="*72)
print("FIGURES")
print("="*72)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE_RAMP = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
    'font.size': 9,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_2,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': MUTED,
    'ytick.labelcolor': MUTED,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
})

def tidy(ax, ylabel, xlabel):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xlabel(xlabel, color=INK_2)

# --- Figure 1: KNN hyperparameter sweep ---
fig, ax = plt.subplots(figsize=(6.3, 3.6))
uni = [(r[0], r[2]) for r in knn_sweep if r[1] == 'uniform']
dis = [(r[0], r[2]) for r in knn_sweep if r[1] == 'distance']
ax.plot(*zip(*dis), color=ORANGE, lw=2, marker='s', ms=5.5, zorder=3,
        markeredgecolor=SURFACE, markeredgewidth=1.2, label="weights='distance'")
ax.plot(*zip(*uni), color=BLUE, lw=2, marker='o', ms=6, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1.2, label="weights='uniform'")

ax.axvline(CHOSEN_K, color=AXIS, lw=1, ls=':', zorder=1)
ax.annotate(f'chosen k={CHOSEN_K}, {dict(uni)[CHOSEN_K]:.4f}',
            xy=(CHOSEN_K, dict(uni)[CHOSEN_K]), xytext=(CHOSEN_K - 1.5, 0.671),
            color=INK, ha='right', fontsize=8.5,
            arrowprops=dict(arrowstyle='-', color=AXIS, lw=1))
ax.annotate(f'default k=5\n{dict(uni)[5]:.4f}',
            xy=(5, dict(uni)[5]), xytext=(6.5, 0.600),
            color=MUTED, fontsize=8.5,
            arrowprops=dict(arrowstyle='-', color=AXIS, lw=1))
tidy(ax, '5-fold CV macro-F1', 'n_neighbors (k)')
ax.set_ylim(0.592, 0.678)
ax.legend(loc='lower right', fontsize=8.5, labelcolor=INK_2)
fig.tight_layout()
fig.savefig('figures/fig1_knn_sweep.png', dpi=200)
plt.close(fig)
print("  figures/fig1_knn_sweep.png")

# --- Figure 2: Decision Tree depth sweep, overfitting gap ---
fig, ax = plt.subplots(figsize=(6.3, 3.6))
UNCONSTRAINED_DEPTH = 43
xs = [UNCONSTRAINED_DEPTH if r[0] is None else r[0] for r in tree_sweep]
cv_ys = [r[1] for r in tree_sweep]
tr_ys = [r[3] for r in tree_sweep]

ax.fill_between(xs, cv_ys, tr_ys, color=ORANGE, alpha=0.12, zorder=1)
ax.plot(xs, tr_ys, color=ORANGE, lw=2, marker='s', ms=5.5, zorder=3,
        markeredgecolor=SURFACE, markeredgewidth=1.2, label='Training macro-F1')
ax.plot(xs, cv_ys, color=BLUE, lw=2, marker='o', ms=6, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1.2, label='5-fold CV macro-F1')

ax.axvline(CHOSEN_DEPTH, color=AXIS, lw=1, ls=':', zorder=1)
ax.annotate(f'chosen depth={CHOSEN_DEPTH}\nCV {best_d[1]:.4f}, gap {best_d[4]:.4f}',
            xy=(CHOSEN_DEPTH, best_d[1]), xytext=(13, 0.565),
            color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle='-', color=AXIS, lw=1))
ax.annotate(f'default (None):\ngap {default_tree[4]:.4f}',
            xy=(UNCONSTRAINED_DEPTH, 0.80), xytext=(UNCONSTRAINED_DEPTH - 1, 0.845),
            color=MUTED, fontsize=8.5, ha='right')
tidy(ax, 'Macro-F1', 'max_depth  (None plotted at its realised depth, 43)')
ax.set_ylim(0.55, 1.02)
ax.legend(loc='upper left', fontsize=8.5, labelcolor=INK_2)
fig.tight_layout()
fig.savefig('figures/fig2_tree_depth_sweep.png', dpi=200)
plt.close(fig)
print("  figures/fig2_tree_depth_sweep.png")

# --- Figure 3: confusion matrices ---
cm_models = [(f'KNN (k={CHOSEN_K})', knn_final.predict(X_test)),
             (f'Decision Tree (depth={CHOSEN_DEPTH})', tree_final.predict(X_test))]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
for ax, (name, preds) in zip(axes, cm_models):
    cm = confusion_matrix(y_test, preds, labels=LABELS)
    ax.imshow(cm, cmap=BLUE_RAMP, vmin=0, vmax=cm.max())
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel('Predicted band', color=INK_2)
    ax.set_title(name, color=INK, fontsize=9, pad=8)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            frac = cm[i, j] / cm.max()
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=9.5,
                    color='#ffffff' if frac > 0.55 else INK)
axes[0].set_ylabel('Actual band', color=INK_2)
fig.tight_layout()
fig.savefig('figures/fig3_confusion_matrices.png', dpi=200)
plt.close(fig)
print("  figures/fig3_confusion_matrices.png")

# --- Figure 4: tree feature importances ---
fig, ax = plt.subplots(figsize=(6.3, 4.0))
ordered = importances.sort_values()
ax.barh(range(len(ordered)), ordered.values, color=BLUE, height=0.68, zorder=3)
ax.set_yticks(range(len(ordered)), ordered.index)
for i, v in enumerate(ordered.values):
    ax.text(v + 0.008, i, f'{v:.4f}', va='center', fontsize=8.5,
            color=INK if v > 0.03 else MUTED)
tidy(ax, '', 'Gini importance')
ax.grid(axis='y', visible=False)
ax.grid(axis='x', visible=True)
ax.set_xlim(0, 0.55)
fig.tight_layout()
fig.savefig('figures/fig4_feature_importance.png', dpi=200)
plt.close(fig)
print("  figures/fig4_feature_importance.png")
