import pandas as pd, numpy as np

df = pd.read_csv('data/listingsA2.csv', low_memory=False)   # Raw Inside Airbnb file (Amendment #1)

# Baseline: price -> numeric (needed before any price analysis)
def parse_price_correct(price):
    if pd.isna(price):
        return np.nan
    cleaned_price = str(price).replace('$', '').replace(',', '')
    try:
        return float(cleaned_price)
    except ValueError:
        return np.nan

df['price'] = df['price'].apply(parse_price_correct)

# ---- STEP 1: missing review_scores_rating ----
n_null = df['review_scores_rating'].isna().sum()
print(f"null ratings: {n_null} ({n_null/len(df)*100:.1f}%)")
# Confirm MNAR: every null rating is a 0-review listing
print("all nulls are 0-review:",
      (df.loc[df['review_scores_rating'].isna(), 'number_of_reviews'] == 0).all())

# Flag keeps 0-review rows usable for the price-VARIATION half of the RQ
df['is_reviewed'] = df['review_scores_rating'].notna()
# But rating-based analysis uses only genuine ratings (drop, not impute)
df_rating = df[df['is_reviewed']].copy()
print("rows:", len(df), "-> rating subset:", len(df_rating))

# Impact: drop vs impute on downstream correlation
paired = df.dropna(subset=['price']) # Only keep rows with a price
drop_sp = df_rating['price'].corr(df_rating['review_scores_rating'], method='spearman') # Price vs rating on valid rows only
imp = df['review_scores_rating'].fillna(df['review_scores_rating'].median()) # Impute with median method
impute_sp = paired['price'].corr(imp.loc[paired.index], method='spearman') # Price vs rating (nulls imputed with median)
print(f"Spearman price~rating | drop {drop_sp:.4f} | impute {impute_sp:.4f}") # Print results for both methods of dealing with NaN values


# ---- STEP 2: distance from Melbourne CBD ----
CBD_LAT, CBD_LON = -37.8136, 144.9631 # Flinders St, Melbourne CBD

def haversine_km(lat, lon, clat=CBD_LAT, clon=CBD_LON):
    """Distance (km) from each listing to the CBD via haversine (accounts for Earth's curvature)."""
    R = 6371.0 # Earth radius, km
    dlat = np.radians(clat - lat)
    dlon = np.radians(clon - lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(clat))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a)) # Great-circle distance to CBD

# Continuous feature
df['distance_from_cbd_km'] = haversine_km(df['latitude'], df['longitude'])

# Discretised bands (for description / plots)
df['cbd_band'] = pd.cut(df['distance_from_cbd_km'], bins=[-0.1, 5, 15, np.inf],
                        labels=['inner', 'middle', 'outer'])

# Impact: band sizes + price by band + price~distance correlation
print("distance km min/median/max:",
      round(df['distance_from_cbd_km'].min(),2),
      round(df['distance_from_cbd_km'].median(),2),
      round(df['distance_from_cbd_km'].max(),2))
dist_sp = df['price'].corr(df['distance_from_cbd_km'], method='spearman') # Spearman correlation between price and continuous distance
print(f"Spearman price~distance: {dist_sp:.4f}")

# Make and print a summary of the distance-from-CBD preprocessing step
cbd_band_df = pd.DataFrame()
cbd_band_df['number_of_listings'] = df.groupby('cbd_band', observed=True).size()
cbd_band_df['median_price'] = df.groupby('cbd_band', observed=True)['price'].median()
cbd_band_df.index.name = None
print(cbd_band_df)


# ---- STEP 3: property-type consolidation ----
# 82 raw categories (41 with <10 listings) -> 5 dwelling-type groups
# Room_type already captures entire/private/shared, so we group by DWELLING type
def group_property_type(raw_type):
    """Map raw property_type to a dwelling-type group."""
    type_lower = str(raw_type).lower()
    if any(keyword in type_lower for keyword in
           ['hotel', 'hostel', 'guesthouse', 'guest suite', 'bed and breakfast', 'resort']):
        return 'Hotel/B&B'
    if 'townhouse' in type_lower:
        return 'Townhouse'
    if any(keyword in type_lower for keyword in
           ['rental unit', 'condo', 'serviced apartment', 'loft', 'apartment']):
        return 'Apartment/unit'
    if any(keyword in type_lower for keyword in
           ['home', 'house', 'cottage', 'cabin', 'villa', 'bungalow', 'tiny', 'farm', 'vacation', 'chalet', 'earthen']):
        return 'House'
    return 'Other'

df['property_group'] = df['property_type'].apply(group_property_type)

# Impact: category collapse + rare-category share + price by group
raw_counts = df['property_type'].value_counts()
n_rare = (raw_counts < 10).sum()
rare_share = raw_counts[raw_counts < 10].sum() / len(df) * 100
print(f"BEFORE {len(raw_counts)} categories | {n_rare} have <10 listings ({rare_share:.2f}% of rows)")
print(f"AFTER {df['property_group'].nunique()} groups")

# Make and print a summary of the property-type consolidation preprocessing step
property_type_df = pd.DataFrame()
property_type_df['number_of_listings'] = df.groupby('property_group').size()
property_type_df['median_price'] = df.groupby('property_group')['price'].median()
property_type_df.index.name = None
print(property_type_df)


# ---- Refresh rated subset ----
# df_rating was created in Step 1, before Steps 2-3 added 'distance_from_cbd_km',
# 'cbd_band' and 'property_group'. Rebuild it so the rated subset has every engineered column.
df_rating = df[df['is_reviewed']].copy()
print("rating subset rows:", len(df_rating), "| has new columns:",
      {'distance_from_cbd_km', 'cbd_band', 'property_group'}.issubset(df_rating.columns))
