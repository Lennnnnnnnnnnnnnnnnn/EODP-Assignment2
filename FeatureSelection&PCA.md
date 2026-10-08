# Section 2.4, 2.5, 3.4, 3.5, 4.4, 4.5, 5.4, 5.5

**Owner:** Teck (Ben)
**For:** Sabir (report)

## 2.4 & 3.4, 4.4 Feature Selection

### Filter Method - Mutual Information
This utilises Mutual Information, which analyses the "entropy" (uncertainty) between a target column, price in this scenario as it is being investigated, and the other features in given in preprocessing in order to ascertain the entropy between each pair. This allows inference in the association between each pair, with a higher number indicating more of an association.
This means these features should be investigated further for correlation

The top 3 ended up being: "accommodates: 0.403054", "bedrooms: 0.313188", "room_type_Entire home/apt: 0.292400"
*Note: room_type was converted to numerical data instead of categorical for the library method to function, which results in each of the five room types being assessed as a feature independently*

These scores indicate that pricing is mainly dependent on the number of guests a place holds and the size of the property. Linking back to the question, this suggests ratings do not play an a measurable part.

However, there is a suspicion that this is flawed. "accommodates" and "bedrooms" essentially account for the same thing, and from the data it shows their correlation is at 0.8447, which states they are very correlated
With an assumption that the number of rooms is the primary driver of pricing, both "accomodates" and "bedrooms" are most likely linked to pricing and thus will have high MI scores. This results in redundancy, which means a better feature to measure may be hiding at #4.

### Embedded Method - Decision Trees (Same as Supervised)
The result for the top 3 for Decision Tree is:
`accommodates`, `room_type_Entire home/apt`, `bedrooms`

This aligns with what is seen in the filter. This may suggest that these are indeed the top three. However, it could fall to the same collinearity trap discussed in the previous part. 

I tuned max_depth over 12 values using 5-fold stratified cross-validation, and only on the training set. Depth 8 won, with a CV macro-F1 of 0.6604.

The default tree with no depth limit grows to depth 43 and 4,533 leaves. It scores 0.9990 on the training data but only 0.5962 in CV, a gap of 0.4028. At depth 8 that gap shrinks to 0.0382.

Feature importance here is the total Gini impurity reduction across every node that splits on that feature, weighted by how many listings reach those nodes, then normalised so everything sums to 1. That means each feature is scored given the other columns. Mutual information is different because it scores each feature on its own.

One bias to be aware of is cardinality. A feature with more distinct values has more candidate thresholds to try, so it gets more chances to find a good split. distance_from_cbd_km has 15,092 distinct values, while a one-hot column has only 2.

### Discrepancies
The two methods show the same top 3, but not in the correct order.
MI has
1. accomodates 
2. bedrooms
3. room_type_Entire home/apt

while Decision Tree:
1. accomodates
2. room_type_Entire home/apt
3. bedrooms

"bedrooms" is valued much less in the decision tree compared to MI. This is the main difference between the 2.
Decision trees are less susceptible to collinearity bias as once they split with a feature, any feature that is correlated is essentially useless in splitting the tree, compared to filter methods which take them independently, which could explain why in MI "bedrooms" has a score of 0.313188 compared to the decision tree's 0.1713. 

There are a few possible explanations to this scenario of the same top 3
1. This is indeed the top 3 features for the data set. This can happen with some data.
2. There exists some bias present in the dataset that results in these 3 appearing at the top
3. The discrepancies appear past the top 3. This is seen as #4 for MI is room_type_Private room at 0.272219 compared to Decision Trees' distance_from_cbd_km 0.0844. This also indicates distance may be a variable to investigate

### Hard Case
A potential hard case is 
id  price  accommodates  distance_from_cbd_km  review_scores_rating        room_type
4274  28207707  177.0             6             37.005183                  4.76  Entire home/apt

The filter method and Decision Trees indicates accomodates and the room_type drive pricing in AirBnB. However, this property is unusally cheap for a whole house accomodating 6 people. This disagrees with what is shown later where the average price per person sits around ($60-70)
Mutual Information and Decision Trees here would indicate that the accommodates would drive this pricing, as 6 people and booking big properties would lead to a hypothetical "discount". However, other factors may be more likely at play, most likely the distance from the CBD as no one wants to pay big for a property far away from the central entertainment district.

### Limitations
- Handling of outlier could be better
- Top 3 may hide or alienate features which are important
- Collinearity is not handled 
- Data is not of the same typing, which resulted in needing discretisation


## 2.5 & 3.5, 4.5 K-means, Hierarchical, PCA, Dimenisionality reduction

### Possible Sets
Set 1:
price
accommodates 
bedrooms
room_type Entire home/apt

Set 2:
price
accommodates
distance_from_cbd_km
room_type Entire home/apt

Set 3:
price 
review_scores_rating
distance_from_cbd_km
accommodates

Settled on Set 2 as with "accomodates" and "bedrooms" shown to be correlated, avoiding any biases in the clustering is a good idea.
Avoiding reviews_scores_rating as the ceiling problem since the median rating is 4.86, most data will be around 4.8 - 5 for rating, which distorts the clustering of the data, along with visualisation.
If the rest of the data can be clustered around the other factors, it may also show that ratings is not the primary driver of pricing.

### K-means & Hierarchical
With VAT visualisation, it suggests that there are 3-4 clusters (k = 3 or 4) in the dataset with the number of squares along the diagonal. The elbow method shows visually that 3 sits better in the elbow compared to 4, with the change going from 3 to 4 being more significant than 4 to 5

After K-means and Hierarchical clustering, the three clusters are:
Blue for K-means and Red for Hierarchical: This is this the most distinct cluster, very isolated from the other clusters. Has some outliers but mainly clustered around the centroid. Some overlap with the next cluster
This is likely the group of small AirBnBs with low accomodates in the data set, some near some far away. Pricing slowly increases as distance increase, along with the size of the property
Green in K-means, Blue for Hierarchical: This cluster is the most tightly packed around its centroid. Sharp diagonal separation between this and the next cluster. Overlap with the previous cluster. Given density, assuming a lot of the data sits here.
This is most likely the main part of AirBnB, small accomodates, non-homes and they show that pricing is mainly based on the number of people it can hold. However pricing does actually shrink as distance increase.
The final cluster, Red on K-means, Green on Hierarchical is the most spread out from its centroid. These are the expensive AirBnB. They represent the high-end properties. The relationship between distance and pricing is not so apparent here. There is not a meaningful enough pattern to discern here, however the further a property is from the cbd may indicate more accommodates.

These clusters do indicate that there exists a meaningful relationship between pricing accomodates. It's hard to ascertain for distance and pricing. It doesn't help much in terms of narrowing down the relationship between ratings and pricing.

Both methods produced clusters which were very similar. Aside from a few exceptions along the diagonal that divides the two clusters, both methods provided the same clustering. This is most likely due to the room_type being a binary seperator and pricing being distinct enough to allow good clustering.


### PCA & Stats
PCA Loadings:
                                 PC1       PC2
price                      0.610880  0.030638
accommodates               0.592923  0.137314
distance_from_cbd_km       0.076133  0.933507
room_type_Entire home/apt  0.519106 -0.329805

Variance explained by PC1: 50.8%
Variance explained by PC2: 26.4%

Average Property Profile per Cluster
                     price  accommodates  distance_from_cbd_km  room_type_Entire home/apt
kmeans_cluster                                                                           
0               466.767813      6.992429             24.121358                   0.988795
1               123.794200      1.941945             13.455141                   0.000000
2               256.693116      3.585997              5.380635                   1.000000

Average Property Profile per Hierarchical Cluster
                           price  accommodates  distance_from_cbd_km  room_type_Entire home/apt
hierarchical_cluster                                                                           
1                     127.102830      1.966571             13.472068                   0.000000
2                     251.101028      3.648899              4.893076                   1.000000
3                     451.298920      6.404819             22.838566                   0.996823


The first PCA1 suggests a "size and cost axis" where the two are linked (larger place = more expensive), while PCA2 is entirely dominated by distance_from_cbd_km, suggesting a "distance axis" 
Price barely appears in PCA2, suggesting distance has little to do with pricing.

This data seems to show there is a baseline pricing per person of a property ($60-70), and that it is unaffected or not measurably affected by other factors

Given 77.2% of the variance is explain by the feature set, this is an ok feature set to use

### Limitations & Improvements
- Reviews suffered major ceiling problem - add log scale or normalisation
- Limiting to 4 means not being able to add reviews to the clustering and PCA
- Feature set only explains 77% of variance in 2D, more data maybe hiding in other features or limitation of 2D
- 3D plot may provide better visualisation of clusters, reducing overlapping
- PCA only records linear associations.