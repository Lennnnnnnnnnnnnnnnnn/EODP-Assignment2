# Section 2.3 / 3.3 / 4.3 / 5.3 - Supervised Learning: Working Notes

**Owner:** Ten
**For:** Sabir (report writing/editing), plus Ben for the feature list
**Status:** Modelling done. Figures done. Notebook and README still to come.

---

# STEP SUMMARY

## Price bands
We started off by splitting the bands into three equal chunks (See Decision log 3) rather than
splitting them based on skewness of the data. The cut points came out at $194.25 and $307.00,
and each band ended up with about 5,460 listings.

## Encoding the features
Look at each row, excluding column that can cause any leakage and have relationship with the
price. Text columns had to become numbers before sklearn would take them, so `room_type` and
`property_group` got one-hot expanded. Ended up with 13 feature columns from 6 original ones.

## Train/Test split
Split the training/test data by 80/20%, stratified so all three bands keep the same share in
both halves. 13,100 rows to train on, 3,275 held back.

## KNN + Decision Tree implementation and optimisation
Run KNN through various different value to find the most optimal CV folding sweep and Accuracy,
same with DT. Tried 11 values of k across two weighting schemes, and 12 tree depths. Best came
out at k=25 uniform and depth 8.

## KNN + Decision Tree Visualisation and Evaluation
Refit both at those settings, scored them once on the held-out test set, then made four figures.
Both land around 0.66 macro-F1, which is double the 0.3337 baseline. The two models are
statistically tied once you look at the confidence intervals.

---

## 2. Decision log

### Decision 1: Classification, not regression

**Chose** to predict a price tier instead of the dollar value.

The spec forces this one. Section 3.4 wants per-class precision/recall/F1 and a majority-class
baseline, and neither of those exists for a continuous target. There's no "precision" for a
predicted $187.43.

We did consider regression on raw price with RMSE. Besides failing the required metrics, price is
badly right-skewed here (median $247.84, max $50,093.56), so RMSE would have been dominated by a
handful of extreme listings.

### Decision 2: Rated listings only, drop rather than impute

**Chose** to model on `df_rating`, the listings that actually have a `review_scores_rating`.

Rating is central to our RQ, and Sabir already showed the missingness is MNAR: all 4,477 null
ratings (17.4% of listings) belong to listings with zero reviews. Putting a median rating on a
listing nobody has reviewed would invent evidence of "perceived value" that isn't there, and it
would bias the exact relationship we're testing.

This also matches what 2.1 already did, so our sections don't contradict each other.

The alternative was keeping all 25,728 rows with an imputed median. Sabir measured it: Spearman
price~rating is 0.0921 dropping versus 0.0796 imputing, so imputing flattens an already weak
relationship by roughly 14%.

The cost is row attrition, which is our main 5.3 limitation.

### Decision 3: Tertiles via `pd.qcut`, not fixed dollar thresholds

**Chose** three equal-count bands.

Equal counts push the majority baseline down to 0.3337, which keeps the baseline comparison
honest. If we'd used round numbers like $100/$250, the skew would have given us very lopsided
classes, and then a model that always guessed the biggest class would post a decent-looking
accuracy while learning nothing.

We still get the "meaningful" reading for free by reporting where the cut points landed. The
boundaries at **$194.25 and $307.00** are themselves an answer to the "what variation in pricing
exists" half of the RQ.

Three bands and not four because it keeps the confusion matrix 3x3 and the per-class table three
rows, which matters at a 10-page limit. It also lines up with the existing 3-level `cbd_band`
from 2.1.

### Decision 4: Raw `price` kept out of the features

The bands were cut from `price`, so leaving it in would let either model read the answer straight
off and report a meaningless near-100%. Classic target leakage.

Same reasoning got rid of two more:
- `cbd_band`, because it was cut from `distance_from_cbd_km` and we already have that
- `review_scores_value`, which is subtler. Guests rate "value for money" as a reaction to the
  price they paid, so it partly encodes the thing we're predicting.

`instant_bookable` went too, but only because it's 100% null in our subset.

### Decision 5: `bedrooms` imputed within `room_type`

`bedrooms` was 14.1% null, which was too much to ignore and too useful to drop. Before picking a
fill I checked whether the nulls were random, and they're clearly not:

```
Private room     1905 / 3337  (57.1% null)
Shared room        61 /  115  (53.0%)
Hotel room          4 /   40  (10.0%)
Entire home/apt   347 / 12883  (2.7%)
```

A private room listing doesn't report a bedroom count because the listing *is* one room. So the
missingness is structural, and filling with the median **within each `room_type`** makes far more
sense than one global median. The spec actually names this exact approach in its preprocessing
examples ("justify your approach with reference to `room_type`").

### Decision 6: Kept both `accommodates` and `bedrooms` even though they correlate at 0.8643

Spearman between the two is 0.8643 (Len's 2.2 table, n = 14,058, the rows where both are
present). I originally measured 0.8526 on the wider rated subset before dropping unpriced rows,
so use Len's figure everywhere since his is computed on the same listings we model on. Spec 2.2
does suggest you might "drop one of two strongly
related predictors before modelling". But that rule exists for linear models, where two nearly
identical columns make the coefficients unstable. Trees and KNN don't fit coefficients, so the
problem doesn't transfer.

Rather than argue it, I tested all three options with 5-fold CV on the training set:

| Feature set | KNN macro-F1 | Tree macro-F1 |
|---|---|---|
| keep both (13 cols) | **0.6239** | **0.5962** |
| drop bedrooms (12) | 0.6087 | 0.5851 |
| drop accommodates (12) | 0.6123 | 0.5931 |

Dropping either one costs us. Keeping both wins for both models, so that's what we did.

The tree barely cares which of the two it has (losing `accommodates` costs it only 0.0031, inside
the fold noise of 0.0074), but KNN clearly wants both (0.0116). KNN needs the extra dimension to
tell listings apart, whereas the tree just picks whichever column splits best and moves on.

### Decision 7: `weights='uniform'` for KNN, not `'distance'`

I tried distance weighting expecting it to help, and it didn't. It loses 0.0076 at the best
setting and is behind uniform at every k above 5.

Best guess at why: 9 of our 13 columns are one-hot binaries, so a lot of listings sit at almost
identical positions. Weighting votes by 1/distance then hands huge influence to a few
near-duplicate neighbours, which partly recreates the k=1 problem. Since a large k is exactly
what was helping us, distance weighting works against it.

### Decision 8: k=25 rather than k=31

They tie at 0.6644. k=25 has lower fold variance though (0.0117 versus 0.0173), so it's the more
stable choice, and it's a slightly simpler model. Same score, less wobble.

### Note on how the hyperparameter grids were picked

why these values: both grids include the sklearn default (k=5, depth=None) so we
can state the effect of changing it, they're dense where the curve actually moves and sparse
where it doesn't, and crucially they extend past the peak. KNN peaks at 25 to 31 then drops off
at 41, and the tree peaks at 8 then declines. If the best value had been the largest one tested
we wouldn't know whether something bigger was better.

---

## 3. Number Comp (single source of truth)

Any figure below that also appears in another section must match. Please pull from this table
rather than re-deriving.

| Quantity | Value |
|---|---|
| All listings in dataset | 25,728 |
| Listings with a rating (`is_reviewed`) | 21,251 |
| Listings with null rating (MNAR, 0 reviews) | 4,477 (17.4%) |
| Rated **and** priced (final modelling set) | **16,375** |
| Rows lost to null price within rated subset | 4,876 |
| Total attrition from full dataset | 9,353 rows (36.4%) |
| Training rows | 13,100 |
| Test rows | 3,275 |
| Feature columns after encoding | 13 |

**Price distribution of the modelling set**

| Quantity | Value |
|---|---|
| Minimum | $7.48 |
| Median | $247.84 |
| Maximum | $50,093.56 |
| Listings above $1,000 | 271 |
| Listings at $0 | 0 |

**Price band definition and balance**

| Band | Range | Count | Share |
|---|---|---|---|
| low | $7.48 - $194.25 | 5,460 | 33.3% |
| mid | $194.25 - $307.00 | 5,465 | 33.4% |
| high | $307.00 - $50,093.56 | 5,450 | 33.3% |

**Majority-class (0R) baseline accuracy: 0.3337** (macro-F1 0.1668). Everything gets compared
against this.

One thing on the split: stratification held to four decimals. The band shares are 0.3334 /
0.3337 / 0.3328 in the full set, the training set and the test set alike, so no band got
unlucky.

---

## 4. Results

### Before and after tuning

| Model | Accuracy | Macro-F1 | Accuracy gain over baseline |
|---|---|---|---|
| Majority baseline (0R) | 0.3337 | 0.1668 | - |
| KNN, default k=5 | 0.6250 | 0.6232 | +0.2913 |
| Tree, default (unpruned) | 0.5966 | 0.5969 | +0.2629 |
| **KNN, tuned k=25** | **0.6568** | **0.6603** | **+0.3231** |
| **Tree, tuned depth=8** | **0.6605** | **0.6661** | **+0.3267** |

Tuning bought the tree more than it bought KNN, which makes sense given how badly the unpruned
tree was overfitting.

### Hyperparameter justification (the three things 3.4 asks for)

| Hyperparameter | Default | Chosen | Effect on CV macro-F1 |
|---|---|---|---|
| `n_neighbors` | 5 | 25 | +0.0405 |
| `weights` | uniform | uniform (kept) | distance tested, -0.0076, rejected |
| `max_depth` | None | 8 | +0.0642 |

The tree's overfitting gap (train minus CV) drops from **0.4028** at the default to **0.0382** at
depth 8.

### Per-class breakdown, test set

| Band | KNN precision | KNN recall | KNN F1 | Tree precision | Tree recall | Tree F1 |
|---|---|---|---|---|---|---|
| low | 0.7669 | 0.7170 | 0.7411 | 0.7989 | 0.6767 | 0.7328 |
| mid | 0.5252 | 0.6002 | 0.5602 | 0.5238 | 0.6734 | 0.5893 |
| high | 0.7085 | 0.6532 | 0.6797 | 0.7280 | 0.6312 | 0.6762 |

`mid` is clearly the hard one for both models, and the reason is structural rather than a flaw in
the models. `mid` is bounded on both sides, so it collects errors from both directions, while
`low` and `high` each only have one neighbouring band to be confused with.

The confusion matrices back this up. The tree pushes 312 actual-`low` and 357 actual-`high`
listings into `mid`. Meanwhile `low` and `high` almost never get mixed up with each other (41 and
45 cases). Neither model was told the bands are ordered, and they still learned that ordering.

### Uncertainty

2,000 bootstrap resamples of the test set, 95% interval on macro-F1:

| Model | Macro-F1 | 95% CI |
|---|---|---|
| KNN (k=25) | 0.6603 | [0.6441, 0.6772] |
| Tree (depth=8) | 0.6662 | [0.6505, 0.6818] |
| Baseline | 0.1668 | [0.1610, 0.1729] |

Those two intervals overlap heavily, so the tree does not beat KNN. The 0.0058 difference sits
inside the noise, and the two models are statistically indistinguishable on this data.

### How depth 8 was chosen, and how the importances are produced

**Choosing depth 8.** `max_depth` was swept over 12 values (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25,
None) with 5-fold stratified cross-validation on the training set only, scoring macro-F1. Depth 8
won at CV macro-F1 0.6604. The test set played no part in choosing it, otherwise the final
reported number would not be an honest estimate of unseen performance.

What makes depth 8 the right stopping point is the gap between training and CV score:

| max_depth | train macro-F1 | CV macro-F1 | gap |
|---|---|---|---|
| 2 | 0.6370 | 0.6370 | 0.0000 |
| 8 | 0.6986 | 0.6604 | 0.0382 |
| 20 | 0.9155 | 0.6077 | 0.3078 |
| None (reaches 43) | 0.9990 | 0.5962 | 0.4028 |

sklearn's default is `max_depth=None`, which here grows to depth 43 with 4,533 leaves. It scores
0.9990 on data it has already seen and 0.5962 on data it has not, which is the tree memorising
individual listings rather than learning the pattern. Depth 8 is where CV peaks while the gap is
still small. Figure 2 plots the whole curve.

**How the importances are produced.** At every node the tree tries each feature and each
threshold and picks the split that most reduces Gini impurity, which measures how mixed the three
bands are within that node. A feature's importance is the total impurity reduction across every
node that split on it, weighted by how many listings reach those nodes, then normalised so the 13
values sum to 1. So it measures how much work a column did separating the bands given the other
columns available, rather than how related it is to price on its own. That is the structural
difference from mutual information, which scores each feature against price in isolation and
never accounts for what the other features already explained.

**Cardinality bias.** A feature with more distinct values offers more candidate thresholds, so it
wins more splits and accumulates more importance. `distance_from_cbd_km` has 15,092 distinct
values across the 16,375 listings; a one-hot column has 2. With no depth limit the tree keeps
splitting to isolate individual rows and reaches for the highest-cardinality column to do it,
which is how an unpruned tree on this data puts distance at 0.4112 and rating at 0.1682. Depth 8
allows only a few hundred splits, so they go to genuinely predictive features.

The bias shrinks at depth 8 but does not disappear. Rating is also continuous (151 distinct
values) and still lands at 0.0370, so the conclusion that rating contributes little holds. Len's
NMI, which has no such bias, puts rating last at 0.0074.

### Feature influence

Decision tree Gini importances, all 13:

| Rank | Feature | Importance |
|---|---|---|
| 1 | accommodates | 0.4774 |
| 2 | room_type_Entire home/apt | 0.1819 |
| 3 | bedrooms | 0.1713 |
| 4 | distance_from_cbd_km | 0.0844 |
| 5 | review_scores_rating | 0.0370 |
| 6 | property_group_Hotel/B&B | 0.0189 |
| 7 | property_group_House | 0.0123 |
| 8 | room_type_Private room | 0.0072 |
| 9 | property_group_Apartment/unit | 0.0041 |
| 10 | room_type_Hotel room | 0.0039 |
| 11 | property_group_Other | 0.0009 |
| 12 | property_group_Townhouse | 0.0007 |
| 13 | room_type_Shared room | 0.0000 |

Ben, your embedded top 3 is `accommodates`, `room_type_Entire home/apt`, `bedrooms`.

Rating drives 3.7% of the tree's decisions. Capacity, exclusivity and bedroom count together
drive about 83%. On the first half of the RQ, price is barely linked to rating. On the second
half, the variation between properties is mostly about physical size and whether you get the
whole place to yourself.

One caveat on this table: Gini importance is biased toward continuous and high-cardinality
features, because `accommodates` has 16 distinct values and therefore 15 candidate split points,
while a one-hot column has one. Rating is continuous too (151 distinct values) and still only
reaches 0.0370, so the conclusion holds, but the ranking between continuous and binary features
isn't a like-for-like comparison.

**Cross-check against Len's 2.2 results.** His NMI against price ranks the same predictors on
the same 16,375 listings, by a completely different method:

| Len's NMI with price | | My tree's Gini importance | |
|---|---|---|---|
| room_type | 0.2246 | accommodates | 0.4774 |
| accommodates | 0.1873 | room_type_Entire home/apt | 0.1819 |
| bedrooms | 0.1746 | bedrooms | 0.1713 |
| property_group | 0.0708 | distance_from_cbd_km | 0.0844 |
| distance_from_cbd_km | 0.0214 | review_scores_rating | 0.0370 |
| review_scores_rating | 0.0074 | | |

Same top three and same bottom, from a filter method and an embedded method that share no
machinery. The one difference is `room_type` ranking first by NMI but second by Gini, which is
the continuous-feature bias above: NMI treats `room_type` as a whole 4-level variable while the
tree splits its importance across four separate one-hot columns.

---

## 5. Why 2.2 and 2.3 give different pictures

Looked at one pair at a time, nothing predicts price well. On the 16,375 listings we model,
Spearman price~rating is 0.0921 and price~distance is only -0.0308. By NMI, rating sits last of
all six predictors at 0.0074.

(Sabir's 2.1 output quotes price~distance as -0.0484 and gives median price by CBD band as inner
$253.00, middle $219.50, outer $247.50. Those are computed on the full 25,728 listings, not our
modelling subset, so don't mix them with the numbers above. Say which population each belongs
to if both appear in the report.)

But both models get to roughly double the baseline. Individually weak predictors turned out to be
jointly informative, and correlation structurally cannot show that because it only ever looks at
two variables at a time.

Len's section and mine are answering different questions rather than contradicting each other.

This doesn't rescue rating, though. The joint signal is coming from the structural features, as
the importance table shows. Perceived value stays weakly related to price while physical
attributes drive it.

---

## 6. Figures

All four are in `figures/`, 200 dpi PNG. Captions go below each one at 9pt, and you'll need to
write those. I left titles off figures 1, 2 and 4 on purpose so the caption doesn't repeat text
already inside the image. Figure 3 keeps its two panel titles because they're what tells the two
matrices apart.

| File | What it shows | Where it belongs |
|---|---|---|
| `fig1_knn_sweep.png` | CV macro-F1 across all 11 k values, uniform versus distance | 3.3, next to the hyperparameter table |
| `fig2_tree_depth_sweep.png` | Training versus CV macro-F1 across depths, with the gap shaded | 3.3 and 4.3 |
| `fig3_confusion_matrices.png` | Both models' confusion matrices side by side | 3.3, next to the per-class table |
| `fig4_feature_importance.png` | All 13 Gini importances, sorted | 3.3, 4.3 and 4.4 |

Notes on each:

**Figure 1** covers the "report the score at every hyperparameter value tried" requirement
visually, so it's satisfied in both the table and the figure. The two lines also make the
rejected `weights='distance'` alternative visible rather than just asserted.

**Figure 2** shows the overfitting directly. The shaded area between the two lines is the gap
between training and CV performance, and it opens up as depth increases.

**Figure 3** shows the `mid` problem more clearly than the numbers do. The middle column is dark
in both panels while the two corners (low predicted as high, high predicted as low) stay pale.

**Figure 4** puts `accommodates` at 0.4774 against `review_scores_rating` at 0.0370.

On colours: the blue and orange are a colourblind-safe pair, and each series also has its own
marker shape, so the figures still read if the report gets printed in greyscale.

---

## 7. Limitations for 5.3

- **36.4% row attrition.** We model 16,375 of 25,728 listings. Whatever we conclude applies to
  reviewed, priced listings, not the whole Melbourne market. New and unreviewed properties are
  systematically excluded and they may well be priced differently.
- **Band boundaries are arbitrary at the margin.** A $193 listing and a $196 listing are
  effectively the same thing but land in different classes. Errors clustered near $194.25 and
  $307.00 are boundary noise, not real model failure.
- **The top band is very wide.** "high" runs from $307 to $50,093, so it lumps ordinary
  three-bedroom houses in with 271 listings above $1,000. Much less internally consistent than
  the other two bands.
- **`mid` is genuinely hard.** F1 of 0.5602 (KNN) and 0.5893 (tree) against roughly 0.73 for
  `low`. Being bounded on both sides is the structural reason.
- **The two models are not separable.** Overlapping confidence intervals mean they can't be
  ranked on performance, only on interpretability.
- **Gini importance is biased** toward continuous and high-cardinality features, so the ranking
  between continuous and one-hot columns isn't like-for-like.
- **No temporal or occupancy data.** Single snapshot, so seasonality, events and actual booking
  rates are invisible. And `price` is the asking price, not revenue.
- **Rating is self-selected.** Only guests who booked and then chose to review contribute, so it
  measures satisfaction among people already willing to pay that price.
- **`bedrooms` is 14.1% imputed.** Defensible given the missingness is structural, but the
  imputed values are still our construction, and `bedrooms` is the third most important feature
  in the tree.

---

## 8. Where the code lives

- `EODP_A2_Supervised.py` is my 2.3 work. It imports the dataframes from Sabir's preprocessing
  script so there's only one definition of the cleaned data rather than two copies that can drift
  apart.
- `figures/` holds the four PNGs, regenerated every time the script runs.

**To reproduce everything:** put `listingsA2.csv` in `data/`, then run
`python EODP_A2_Supervised.py`. Not the preprocessing file, since importing it runs it
automatically. Needs `pandas`, `scipy`, `scikit-learn` and `matplotlib`.

`code.ipynb` and `README.txt` are still to do. Waiting on Len's and Ben's code before I merge
everything into the one notebook.
