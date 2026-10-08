import pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from EODP_A2_Preprocessing import df_rating
from EODP_A2_Supervised import importances

df_copy = df_rating.dropna(subset=['price']).copy()   # rated + priced listings

# Following the same filling empty bedrooms with median done by Ten so that dat stays consistent 
df_copy['bedrooms']  = df_copy.groupby('room_type')['bedrooms'].transform(lambda x: x.fillna(x.median()))
df_copy['bedrooms']  = df_copy['bedrooms'].fillna(df_copy['bedrooms'].median())   

df_copy = df_copy[df_copy['price'] < 1000]  # remove extreme outliers so that feature selection is more accurate
# One-hot encode categorical variables for mutual information
# Algorithm requires numeric values, so pd.get_dummies convert categorical vars to binary columns

X = df_copy[['review_scores_rating', 'accommodates', 'bedrooms', 'distance_from_cbd_km', 'room_type', 'property_group']]
X = pd.get_dummies(X, columns=['room_type', 'property_group'], drop_first=False)

X = X.fillna(X.median())
y = df_copy['price']

# use built-in mutual_info_regression function to compute mutual information scores
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("Top 3 Filter Features (Mutual Information)\n")
print(mi_series.head(3))

print(mi_series.head(4))

# Decision Tree Regressor
# Same decision tree technique in EODP_A2_Supervised.py to get feature importances

top_3_embedded = importances.sort_values(ascending=False).head(3)
print("Top 3 Embedded Features (Decision Tree)\n")
print(top_3_embedded)

# Below is for report #

# finding a hard case
hard_case = df_copy[(df_copy['distance_from_cbd_km'] < 1.0) & 
                    (df_copy['review_scores_rating'] > 4.8) & 
                    (df_copy['room_type'] == 'Shared room')][['id', 'price', 'accommodates', 'distance_from_cbd_km', 'review_scores_rating', 'room_type']]
print(hard_case.head(1))

second_hard_case = df_copy[(df_copy['accommodates'] >= 6) & 
                      (df_copy['room_type'] == 'Entire home/apt') & 
                      (df_copy['distance_from_cbd_km'] > 35) & 
                      (df_copy['price'] < 200)][['id', 'price', 'accommodates', 'distance_from_cbd_km', 'review_scores_rating', 'room_type']]
print(second_hard_case.head(1))

# correlation between accommodates and bedrooms
correlation = df_copy['accommodates'].corr(df_copy['bedrooms'])
print(f"Correlation for accommodates and bedrooms: {correlation:.4f}")

mean = df_copy['review_scores_rating'].mean()
median = df_copy['review_scores_rating'].median()
percentiles = df_copy['review_scores_rating'].quantile([0.25, 0.50, 0.75, 0.90, 0.95])

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print("\nPercentiles:" + percentiles.to_string())



