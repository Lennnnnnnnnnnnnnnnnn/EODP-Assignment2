import pandas as pd
from EODP_A2_Preprocessing import df, df_rating

print("\n" + "="*60)
print("2.3 SUPERVISED LEARNING - target construction")
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
