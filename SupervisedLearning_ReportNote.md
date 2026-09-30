# Section 2.3 / 3.3 / 4.3 / 5.3 - Supervised Learning: Working Notes

**Owner:** Ten
**For:** Sabir (report writing/editing)
**Status:** Target variable built. Models not yet trained.
**Last updated:** 30 Sep 2026


---

## 1. How 2.3 answers the research question

**RQ:** "Is the pricing of a property linked to its overall rating, and what kinds of variation
in pricing exists between properties?"

Section 2.2 (Len) tests the *association* between price and rating. Section 2.3 tests the same
link *predictively*: given a listing's rating and its other attributes, can a model place that
listing in the correct price tier? If rating carries real information about price, a model with
access to rating should beat a model guessing blindly.

This maps onto two of the spec's own example RQs ("Can listings be grouped into price tiers, and
what attributes distinguish tiers?" and "What listing and location attributes are associated with
price variation... is price associated with perceived value?"), so the framing is sanctioned by
the spec rather than invented by us.

---

## 2. Decision log

### Decision 1: Classification, not regression

**Chosen:** Predict a categorical price *tier*, not the dollar value.

**Why:** Spec 3.4 requires per-class precision/recall/F1 and a majority-class (0R) baseline.
Neither metric exists for a continuous target - there is no "precision" for a predicted $187.43.
Regression would force us to report RMSE/MAE instead and we would fail those bullet points.

**Alternative considered:** Regression on raw price, reporting RMSE. Rejected because it does not
satisfy the required metrics, and because price is right-skewed (median $247.84, max $50,093.56),
so RMSE would be dominated by a handful of extreme listings.

### Decision 2: Rated listings only (drop, not impute)

**Chosen:** Build the model on `df_rating` (listings that have a `review_scores_rating`).

**Why:** Rating is a core feature for this RQ, and Sabir's Step 1 established the missingness is
MNAR - every one of the 4,477 null ratings (17.4% of all listings) belongs to a listing with zero
reviews. Imputing a median rating onto a listing nobody has reviewed would invent evidence of
"perceived value" that does not exist, and would bias the exact relationship we are testing.

**Consistency note:** this matches the drop-not-impute choice already made in 2.1, so the two
sections do not contradict each other.

**Alternative considered:** Keep all 25,728 listings and impute the median rating. Sabir measured
the effect: Spearman price~rating is 0.0921 when dropping versus 0.0796 when imputing, so
imputation flattens an already weak relationship by about 14%.

**Cost of this choice:** see row attrition in section 3 below. This is our main 5.3 limitation.

### Decision 3: Tertiles via `pd.qcut`, not fixed dollar thresholds

**Chosen:** Three equal-count bands (low/mid/high) using quantile binning.

**Why:** Equal-count bands force the majority-class baseline down to 33.4%, which makes the
baseline comparison honest. Melbourne prices are right-skewed, so round-number thresholds such as
$100/$250 would have produced very unequal classes; a model that always guessed the largest class
would then post a high accuracy while learning nothing, and per-class recall for the smallest
class could sit near zero.

**Alternative considered:** Fixed thresholds at market-meaningful round numbers. Rejected for the
imbalance reason above, and because we could not justify any particular dollar cutoff from the
data without it being arbitrary.

**We still get the "meaningful" reading for free** by reporting the computed cut points: the
tertile boundaries land at **$194.25 and $307.00**, which is itself a direct answer to the "what
variation in pricing exists" half of the RQ.

**Why three bands and not four:** keeps the confusion matrix 3x3 and the per-class F1 table three
rows, which matters in a 10-page limit. Also mirrors the existing 3-level `cbd_band` from 2.1, so
the report reads consistently.

### Decision 4: Raw `price` excluded from the features

**Chosen:** `price` is dropped from the feature set.

**Why:** The target was derived from `price` by binning it. Leaving it in would let any model
recover the band perfectly and report near-100% accuracy that means nothing. This is textbook
target leakage.

---

## 3. Computed numbers (single source of truth)

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

**Majority-class (0R) baseline accuracy: 0.3337** - every model gets compared against this.

---

## 4. Result worth flagging early

**This is the most interesting finding in the section, and it is the reason 2.3 exists as well
as 2.2.**

Taken alone, each variable looks weak. Spearman price~rating is only **0.0921**, and
price~distance only **-0.0484**. Median price by CBD band is not even monotonic (inner $253.00,
middle $219.50, outer $247.50), probably because coastal and peninsula holiday rentals sit in the
"outer" band. On correlation evidence alone you would conclude price is barely predictable.

The models say otherwise. Untuned, on the held-out test set:

| Model | Accuracy | Macro-F1 | Accuracy gain over baseline |
|---|---|---|---|
| Majority-class baseline (0R) | 0.3337 | 0.1668 | - |
| KNN (default k=5) | 0.6250 | 0.6232 | **+0.2913** |
| Decision Tree (default, unpruned) | 0.5966 | 0.5969 | **+0.2629** |

Both roughly **double** the baseline before any tuning.

**Why this matters for the RQ:** individually weak predictors can be jointly informative.
Correlation measures variables two at a time, so it could not have revealed this; only a model
using all 13 feature columns at once could. That contrast between 2.2 and 2.3 is a genuine
insight to make in the discussion, not a contradiction to paper over.

**Nuance to preserve:** none of this rescues *rating* as a predictor. The joint signal most
likely comes from the structural features (capacity, bedrooms, room type, distance). The
feature-importance results in 3.4/4.4 will show which, and the honest reading is probably still
"perceived value is weakly related to price; physical attributes drive it." Do not claim the
models prove a strong price-rating link.

**Correction note:** an earlier version of this file predicted the models would land only
modestly above baseline. That prediction was wrong, as the table above shows. Use these numbers.

---

## 5. Still to do (my side)

1. Finalise feature set and encode categoricals (`property_group`, `cbd_band`, `room_type`)
2. Stratified train/test split, justified against the 33/33/33 balance above
3. Stratified k-fold CV sweeping `n_neighbors` (KNN) and `max_depth` (tree), recording the score
   at **every** value tried, since spec 3.4 requires the full sweep, not just the winner
4. For each hyperparameter changed from default: record the sklearn default, our value, and the
   measured effect on the metric (spec 3.4 asks for all three explicitly)
5. Feature scaling for KNN only, and a note on why the tree does not need it
6. Bootstrap confidence interval on macro-F1 for at least one model
7. Confusion matrices and per-class precision/recall/F1
8. Assemble everything into the single `code.ipynb` and write `README.txt`

## 6. Limitations to carry into 5.3

- **36.4% row attrition.** We model 16,375 of 25,728 listings. Results generalise to *reviewed,
  priced* listings, not to the whole Melbourne market. Newly listed and unreviewed properties are
  systematically excluded, and those may be priced differently.
- **Band boundaries are arbitrary at the margin.** A $193 and a $196 listing are effectively
  identical but land in different classes. Misclassifications clustered near $194.25 and $307.00
  should be read as boundary noise, not real model error.
- **The top band is extremely wide.** "high" spans $307 to $50,093, so it mixes ordinary
  three-bedroom houses with 271 outliers above $1,000. The band is far less internally
  homogeneous than the other two.
- **No temporal or occupancy information.** The dataset is a single snapshot, so seasonality,
  events and actual booking rates are invisible. Price is the *asking* price, not revenue.
- **Rating is a weak and self-selected signal.** Only guests who chose to book and then chose to
  review contribute to it, so it measures satisfaction among people already willing to pay that
  price.

---

## 7. Where the code lives

- `EODP_A2_Preprocessing.py` - Sabir's 2.1 work. **Unchanged by me.**
- `EODP_A2_Supervised.py` - my 2.3 work. Imports the dataframes from the preprocessing script so
  there is one definition of the cleaned data, not two.
- Both get folded into the single `code.ipynb` before submission (my job, per the role split).

**Reproducing the numbers in section 3:** place `listingsA2.csv` in `data/`, then run
`python EODP_A2_Supervised.py` (not the preprocessing file - importing it runs it automatically).
Requires `pandas`, `scipy`, `scikit-learn`.
