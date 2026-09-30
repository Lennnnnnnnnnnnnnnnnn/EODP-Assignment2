import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.metrics import mutual_info_score
from EODP_A2_Preprocessing import df_rating   # 2.1 output: rated subset with engineered columns

print("\n" + "="*60)
print("2.2 CORRELATION ANALYSIS")
print("="*60)

# ---- Variable set + measurement level (decides which methods apply) ----
VARS = {
    'price':                'continuous',  # target, kept continuous
    'review_scores_rating': 'continuous',
    'distance_from_cbd_km': 'continuous',
    'accommodates':         'discrete',    # integer count, equal intervals
    'bedrooms':             'discrete',
    'room_type':            'nominal',     # categorical, no order
    'property_group':       'nominal',
}
N_BINS = 5   # equal-frequency bins for numeric variables in MI/NMI

# Rated + priced listings (rating nulls already dropped in 2.1)
base = df_rating.dropna(subset=['price'])
print(f"rated + priced listings: {len(base)}")

# Pairwise deletion: each pair uses every row where BOTH of its variables are present.
# Only bedrooms has gaps, so pairs without bedrooms keep all rated + priced listings.
corr_df = base[list(VARS)].copy()
print("missing values per variable:", corr_df.isna().sum()[lambda s: s > 0].to_dict())
print("price min/median/max:", corr_df['price'].min(), corr_df['price'].median(), corr_df['price'].max())

# Numeric view for Pearson/Spearman: log(price) tames the right skew for Pearson;
# Spearman is rank-based, so the log does not change its value
num = corr_df.copy()
num['price'] = np.log(num['price'])

# Discrete view for MI/NMI: numeric vars -> equal-frequency bins (skew-robust),
# binned once per variable on all its non-missing values so edges are the same in every pair.
# Ties (e.g. many 5.0 ratings) can merge edges, so report the bins actually produced
disc = pd.DataFrame(index=corr_df.index)
for v, kind in VARS.items():
    if kind in ('continuous', 'discrete'):
        binned, edges = pd.qcut(corr_df[v], q=N_BINS, duplicates='drop', retbins=True)
        disc[v] = binned.cat.codes.where(corr_df[v].notna())   # keep NaN (not code -1)
        if v == 'price':
            print("price bin edges for MI ($):", [round(float(e), 2) for e in edges])
    else:
        disc[v] = corr_df[v].astype(str)
print("categories used for MI/NMI:", {v: disc[v].nunique() for v in VARS})


# ---- Pairwise measures ----
def mi_bits(x, y):
    return mutual_info_score(x, y) / np.log(2)   # sklearn uses nats -> convert to bits

def h_bits(x):
    return entropy(pd.Series(x).value_counts(), base=2)

def na_reason(v1, v2):
    """Pearson/Spearman need ordered values; return why not, or None if they apply."""
    for v in (v1, v2):
        if VARS[v] == 'nominal':
            return f"N/A: {v} is nominal (no order)"
    return None

def pair_stats(v1, v2, keep):
    """All four measures for one pair, computed on the rows in `keep`."""
    row = {'var1': v1, 'var2': v2, 'n': int(keep.sum())}
    reason = na_reason(v1, v2)
    if reason is None:
        row['pearson_r'], row['pearson_p'] = pearsonr(num.loc[keep, v1], num.loc[keep, v2])
        row['spearman_rho'], row['spearman_p'] = spearmanr(num.loc[keep, v1], num.loc[keep, v2])
        row['note'] = 'Pearson uses log(price)' if 'price' in (v1, v2) else ''
    else:
        row['pearson_r'] = row['pearson_p'] = row['spearman_rho'] = row['spearman_p'] = np.nan
        row['note'] = 'Pearson & Spearman ' + reason

    # MI/NMI apply to every pair (nominal included); NMI = MI / mean(H(X), H(Y))
    # (arithmetic mean, same as sklearn's default; min would inflate low-entropy room_type)
    x, y = disc.loc[keep, v1], disc.loc[keep, v2]
    mi = mi_bits(x, y)
    row['MI_bits'] = mi
    row['NMI'] = mi / ((h_bits(x) + h_bits(y)) / 2)
    return row

COLS = ['var1', 'var2', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'MI_bits', 'NMI', 'note']

# Main table: pairwise deletion (rows where both variables are present)
results = pd.DataFrame([pair_stats(v1, v2, corr_df[[v1, v2]].notna().all(axis=1))
                        for v1, v2 in combinations(VARS, 2)])[COLS]

pd.set_option('display.width', 200, 'display.max_columns', 20, 'display.max_colwidth', 80)
print(f"\nAll {len(results)} pairs (pairwise n, values used: {sorted(results['n'].unique())}):")
print(results.round(4).to_string(index=False))

# Target pairs, ranked by NMI (the one measure available for every variable)
target = results[(results['var1'] == 'price') | (results['var2'] == 'price')]
print("\nPairs with price, sorted by NMI:")
print(target.sort_values('NMI', ascending=False)
            [['var1', 'var2', 'n', 'pearson_r', 'spearman_rho', 'MI_bits', 'NMI']].round(4).to_string(index=False))

# Like-for-like check: price pairs recomputed on the rows that have bedrooms, so bedrooms
# pairs (n smaller) can be compared with the other predictors on the same listings
common = corr_df.notna().all(axis=1)
same_n = pd.DataFrame([pair_stats('price', v, common) for v in VARS if v != 'price'])[COLS]
print(f"\nPairs with price on the common sample (n = {int(common.sum())}, rows with bedrooms):")
print(same_n.sort_values('NMI', ascending=False)
            [['var1', 'var2', 'n', 'pearson_r', 'spearman_rho', 'MI_bits', 'NMI']].round(4).to_string(index=False))

# Square matrices per method (handy for heatmaps)
for col in ['pearson_r', 'spearman_rho', 'NMI']:
    m = results.pivot(index='var1', columns='var2', values=col)
    print(f"\n{col} matrix:")
    print(m.reindex(index=list(VARS)[:-1], columns=list(VARS)[1:]).round(4).to_string())
