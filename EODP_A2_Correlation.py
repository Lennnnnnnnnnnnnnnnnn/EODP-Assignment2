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


# ---- Figure: Spearman + NMI heatmaps (lower triangle, values in cells) ----
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

LABELS = {'price': 'Price', 'review_scores_rating': 'Rating', 'distance_from_cbd_km': 'Distance to CBD',
          'accommodates': 'Accommodates', 'bedrooms': 'Bedrooms', 'room_type': 'Room type',
          'property_group': 'Property group'}
INK, MUTED, SURFACE, NA_FILL = '#1f1f1e', '#6b6b67', '#fcfcfb', '#f0efec'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_blue_red', ['#1c5cab', '#86b6ef', NA_FILL, '#f0a3a2', '#b83a39'])

def full_matrix(col):
    """Symmetric matrix of one measure from the pairwise results table."""
    m = pd.DataFrame(np.nan, index=list(VARS), columns=list(VARS))
    for _, r in results.iterrows():
        m.loc[r['var1'], r['var2']] = m.loc[r['var2'], r['var1']] = r[col]
    return m

def text_colour(rgb):
    """Dark ink on light cells, white on dark cells (relative luminance)."""
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return INK if lum > 0.45 else 'white'

def heatmap(ax, col, cmap, norm, title):
    m, names = full_matrix(col), list(VARS)
    rows, cols = names[1:], names[:-1]
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols[:i + 1]):
            val = m.loc[rv, cv]
            fill = NA_FILL if np.isnan(val) else cmap(norm(val))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill, edgecolor=SURFACE, lw=2))
            if np.isnan(val):   # method not applicable (nominal variable)
                ax.text(j + .5, i + .5, 'n/a', ha='center', va='center', fontsize=9, color=MUTED)
            else:
                mark = '†' if 'bedrooms' in (rv, cv) else ''
                label = f'{val:.2f}'.replace('-0.00', '0.00')   # no negative zero
                ax.text(j + .5, i + .5, label + mark, ha='center', va='center', fontsize=9,
                        color=text_colour(to_rgb(fill)))
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + .5, [LABELS[c] for c in cols], rotation=40, ha='right')
    ax.set_yticks(np.arange(len(rows)) + .5, [LABELS[r] for r in rows])
    ax.tick_params(length=0, colors=INK, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    cb = ax.figure.colorbar(ScalarMappable(norm, cmap), ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE)
heatmap(ax1, 'spearman_rho', DIV, Normalize(-1, 1), 'Spearman ρ')
heatmap(ax2, 'NMI', SEQ, Normalize(0, 0.5), 'NMI (arithmetic mean)')
fig.text(0.01, 0.01, f"† pairs with bedrooms: n = {results['n'].min():,}; all other pairs: n = {results['n'].max():,}. "
         "n/a: nominal variable (no order).", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/correlation_heatmaps.png', dpi=200, facecolor=SURFACE)
print("\nsaved figures/correlation_heatmaps.png")
