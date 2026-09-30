import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

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

base['price_band'], bins = pd.qcut(base['price'], q=3,
                                   labels=['low', 'mid', 'high'], retbins=True)

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
print("\nSpearman accommodates~bedrooms = 0.8526 on the rated subset, so the two")
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
