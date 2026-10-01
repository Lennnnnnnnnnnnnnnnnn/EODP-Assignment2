# Section 2.2 / 3.2 / 4.2 - Correlation Analysis: Working Notes

**Owner:** Len
**For:** Sabir (report editing), Ten and Ben (cross-section numbers)
**Status:** Code done, figure done. Report text (2.2 / 3.2 / 4.2) not written yet - Len writes it.
**Last updated:** 30 Sep 2026

> These notes are a record of decisions and numbers only. The explanation and interpretation
> in the report are written by Len.

---

# STEP SUMMARY

| Step | What the code does | Result |
|---|---|---|
| Input | `df_rating` from 2.1, then drop rows with no price | 16,375 rated + priced listings |
| Variable set | 7 variables, each tagged continuous / discrete / nominal | 21 pairs |
| Missing values | Pairwise deletion - each pair uses rows where both variables exist | n = 16,375, or 14,058 for pairs with bedrooms |
| Pearson / Spearman | Numeric pairs only; Pearson on log(price), Spearman on raw values | 10 pairs computed, 11 N/A (nominal) |
| MI / NMI | Every pair; numeric variables binned once (equal-frequency, k = 5) | 21 pairs computed |
| Check | Price pairs recomputed on the common 14,058 rows | like-for-like comparison with bedrooms |
| Figure | Spearman + NMI heatmaps | `figures/correlation_heatmaps.png` |

---

## 2. Decision log

### Decision 1: Variable set

| Variable | Type | Role | Methods |
|---|---|---|---|
| `price` | continuous | target | all 4 |
| `review_scores_rating` | continuous | predictor (RQ part 1) | all 4 |
| `distance_from_cbd_km` | continuous | predictor (location) | all 4 |
| `accommodates` | discrete count | predictor (size) | all 4 |
| `bedrooms` | discrete count | predictor (size) | all 4 |
| `room_type` | nominal, 4 levels | predictor (type) | MI, NMI only |
| `property_group` | nominal, 5 levels | predictor (type) | MI, NMI only |

- Excluded: `cbd_band` (derived from distance), raw `property_type` (replaced by `property_group`),
  IDs / text columns.
- `bedrooms` kept on purpose: needed for the accommodates~bedrooms multicollinearity pair.

### Decision 2: Price kept continuous (not Ten's bands)

- Continuous price lets all 4 methods apply to the target; with bands, Pearson is N/A for every
  target pair.
- Price~rating Spearman on continuous price = **0.0921** (matches 2.1). On tertile bands it
  would be 0.0823 (n = 16,375).
- Pearson uses **log(price)**: price skew = 42.35, min $7.48, max $50,093.56.
- Spearman uses raw price (rank-based, log would not change it).

### Decision 3: Pairwise deletion (option b)

- Only `bedrooms` has gaps: 2,317 of 16,375 rows (14.1%).
- Options considered:

| Option | n | price~rating Spearman | Outcome |
|---|---|---|---|
| (a) single n, keep bedrooms | 14,058 all pairs | 0.1148 | rejected - does not match 2.1 / 2.3 |
| **(b) pairwise** | 14,058 bedrooms pairs, 16,375 others | **0.0921** | **chosen** |
| (c) drop bedrooms variable | 16,375 all pairs | 0.0921 | rejected - loses accommodates~bedrooms |

- Within any one pair, all 4 methods use the same rows.
- No imputation of bedrooms in 2.2 (imputing from accommodates would inflate the
  accommodates~bedrooms pair).

### Decision 4: Discretisation for MI / NMI

- Equal-frequency bins (`pd.qcut`), k = 5, `duplicates='drop'`.
- Binned **once per variable** on all its non-missing values; the same edges are reused in every pair.
- Ties merge bins, so the actual k and counts are not all equal:

| Variable | Skew | k used | Bin edges | Count per bin |
|---|---|---|---|---|
| price | 42.35 | 5 | 7.48 / 145.33 / 216.0 / 280.5 / 380.0 / 50,093.56 | 3,276 / 3,277 / 3,273 / 3,280 / 3,269 |
| review_scores_rating | -4.52 | **4** | 1.0 / 4.6 / 4.8 / 4.9 / 5.0 | 3,276 / 3,453 / 3,120 / 6,526 |
| distance_from_cbd_km | 1.85 | 5 | 0.01 / 1.03 / 3.19 / 7.1 / 19.09 / 79.96 | 3,275 each |
| accommodates | 1.59 | 5 | 1 / 2 / 3 / 4 / 6 / 16 | 6,535 / 898 / 4,103 / 2,911 / 1,928 |
| bedrooms | 1.89 | **4** | 0 / 1 / 2 / 3 / 16 | 5,584 / 5,177 / 2,017 / 1,280 |
| room_type | - | 4 | categories as-is | - |
| property_group | - | 5 | categories as-is | - |

- Note: accommodates and bedrooms bins are far from equal-sized because of tied integer values.

### Decision 5: MI in bits, NMI with arithmetic-mean normalisation

- MI: `mutual_info_score / ln 2` (sklearn returns nats).
- NMI = MI / mean(H(X), H(Y)); verified equal to sklearn
  `normalized_mutual_info_score(average_method='arithmetic')`.
- Alternatives compared (pairs with price):

| Price with | NMI min | **NMI mean** | NMI max |
|---|---|---|---|
| room_type | 0.4338 | **0.2246** | 0.1516 |
| accommodates | 0.1989 | **0.1873** | 0.1769 |
| bedrooms | 0.1994 | **0.1746** | 0.1553 |
| property_group | 0.0931 | **0.0708** | 0.0571 |
| distance_from_cbd_km | 0.0214 | **0.0214** | 0.0214 |
| review_scores_rating | 0.0081 | **0.0074** | 0.0067 |

- room_type entropy is low (0.811 bits), so its rank depends on the normalisation:
  1st under min and mean, 3rd under max.

---

## 3. Number Comp (single source of truth)

| Quantity | Value |
|---|---|
| All listings | 25,728 |
| Rated (`df_rating`) | 21,251 |
| Rated + priced | **16,375** |
| Missing bedrooms within rated + priced | 2,317 (14.1%) |
| Rated + priced + bedrooms | **14,058** |
| Pairs | 21 (6 with price, 15 predictor-predictor) |

**Who the 2,317 missing-bedrooms rows are**

| Group | Missing bedrooms |
|---|---|
| Private room | 1,905 / 3,337 (57.1%) |
| Shared room | 61 / 115 (53.0%) |
| Hotel room | 4 / 40 (10.0%) |
| Entire home/apt | 347 / 12,883 (2.7%) |
| Lowest price tertile (<= $194.25) | 2,017 of the 2,317 |

---

## 4. Results

### Pairs with price (main table, sorted by NMI)

| Price with | n | Pearson (log price) | Spearman | MI (bits) | NMI |
|---|---|---|---|---|---|
| room_type | 16,375 | n/a - nominal | n/a - nominal | 0.3519 | 0.2246 |
| accommodates | 16,375 | 0.6116 | 0.6946 | 0.4108 | 0.1873 |
| bedrooms | 14,058 | 0.5543 | 0.6363 | 0.3542 | 0.1746 |
| property_group | 16,375 | n/a - nominal | n/a - nominal | 0.1326 | 0.0708 |
| distance_from_cbd_km | 16,375 | 0.0388 | -0.0308 | 0.0497 | 0.0214 |
| review_scores_rating | 16,375 | 0.1117 | 0.0921 | 0.0156 | 0.0074 |

All Pearson/Spearman p-values < 0.001 for pairs with price (n > 14,000, so p says little about strength).

### Predictor-predictor pairs

| Pair | n | Pearson | Spearman | MI (bits) | NMI |
|---|---|---|---|---|---|
| accommodates~bedrooms | 14,058 | **0.8480** | **0.8643** | 0.9724 | **0.4966** |
| distance~property_group | 16,375 | n/a | n/a | 0.3673 | 0.1961 |
| accommodates~room_type | 16,375 | n/a | n/a | 0.2078 | 0.1445 |
| bedrooms~property_group | 14,058 | n/a | n/a | 0.2235 | 0.1419 |
| room_type~property_group | 16,375 | n/a | n/a | 0.1054 | 0.0943 |
| accommodates~property_group | 16,375 | n/a | n/a | 0.1003 | 0.0575 |
| distance~bedrooms | 14,058 | 0.2474 | 0.1701 | 0.1078 | 0.0526 |
| bedrooms~room_type | 14,058 | n/a | n/a | 0.0607 | 0.0525 |
| distance~room_type | 16,375 | n/a | n/a | 0.0475 | 0.0303 |
| distance~accommodates | 16,375 | 0.1503 | -0.0046 | 0.0634 | 0.0289 |
| rating~room_type | 16,375 | n/a | n/a | 0.0121 | 0.0088 |
| rating~distance | 16,375 | 0.0589 | 0.1226 | 0.0162 | 0.0076 |
| rating~property_group | 16,375 | n/a | n/a | 0.0125 | 0.0075 |
| rating~bedrooms | 14,058 | 0.0154 | 0.0224 | 0.0046 | 0.0025 |
| rating~accommodates | 16,375 | 0.0231 | -0.0086 | 0.0044 | 0.0022 |

Full table with p-values: printed by `EODP_A2_Correlation.py`.

### Pearson vs Spearman disagreements (numbers only)

| Pair | Pearson | Spearman |
|---|---|---|
| price~distance | +0.0388 | -0.0308 |
| distance~accommodates | +0.1503 | -0.0046 |
| rating~distance | 0.0589 | 0.1226 |

### Common-sample check (price pairs on the same 14,058 rows)

| Price with | Spearman (main n) | Spearman (14,058) | NMI (main n) | NMI (14,058) |
|---|---|---|---|---|
| room_type | n/a | n/a | 0.2246 | 0.1316 |
| accommodates | 0.6946 | 0.6374 | 0.1873 | 0.1519 |
| bedrooms | 0.6363 | 0.6363 | 0.1746 | 0.1746 |
| property_group | n/a | n/a | 0.0708 | 0.0664 |
| distance_from_cbd_km | -0.0308 | **+0.0551** | 0.0214 | 0.0157 |
| review_scores_rating | 0.0921 | 0.1148 | 0.0074 | 0.0092 |

- NMI rank of room_type: 1st on 16,375, 3rd on 14,058.
- price~distance changes sign between the two samples.

---

## 5. Cross-section consistency - needs team agreement

| Number | 2.2 (Len) | Elsewhere | Issue |
|---|---|---|---|
| price~rating Spearman | 0.0921 (n = 16,375) | 2.1: 0.0921; 2.3 notes: 0.0921 | matches |
| price~distance Spearman | **-0.0308** (rated + priced, n = 16,375) | 2.1: **-0.0484**; 2.3 notes section 5 cites -0.0484 | different samples: 2.1 uses all priced listings (n = 19,175, incl. unrated) |
| accommodates~bedrooms Spearman | **0.8643** (n = 14,058, no imputation) | 2.3: **0.8526** (bedrooms imputed within room_type, n = 16,375) | different handling of missing bedrooms |
| bedrooms missing values | dropped per pair | 2.3 Decision 5: imputed with room_type median | two treatments of the same variable |

To decide:
- [ ] Sabir: print the n behind each 2.1 correlation (0.0921 uses 16,375; -0.0484 uses 19,175).
- [ ] Which price~distance figure the report quotes, and on which sample.
- [ ] Whether 2.2 and 2.3 should share one bedrooms treatment (drop vs impute within room_type).
- [ ] Ben: if PCA / clustering uses a correlation matrix, a pairwise-n matrix is not guaranteed
      positive semi-definite - compute it on one complete sample instead of reusing this one.

---

## 6. Figures

| File | What it shows | Where it belongs |
|---|---|---|
| `figures/correlation_heatmaps.png` | Left: Spearman rho (diverging, -1 to 1, n/a for nominal). Right: NMI (sequential, 0 to 0.5). Lower triangle, values in cells | 3.2 |

- Cell values rounded to 2 dp; cite 4 dp values from the tables above.
- † marks pairs with bedrooms (n = 14,058); footnote inside the figure states both n.
- NMI colour scale capped at 0.5 (largest value 0.4966).
- No caption in the image - caption to be written by Len.

---

## 7. Limitations for 5.2 (facts only)

- Row attrition: 16,375 of 25,728 listings (rated + priced); 14,058 for bedrooms pairs.
- The 14,058 bedrooms sample under-represents private rooms (57.1% of them missing bedrooms)
  and the lowest price tertile (2,017 of 2,317 dropped rows).
- Pairwise n: bedrooms pairs and non-bedrooms pairs are on different samples.
- Binning: rating and bedrooms reduced to k = 4 by ties; accommodates bins unequal (898 to 6,535).
- NMI ranking of room_type depends on normalisation (min / mean / max) and on the sample.
- Correlation is pairwise only - it does not capture joint effects of several variables.
- Asking price only, single snapshot (same as 2.3).

---

## 8. Where the code lives

- `EODP_A2_Correlation.py` - imports `df_rating` from `EODP_A2_Preprocessing.py`. Does not
  import anything from 2.3.
- `figures/correlation_heatmaps.png` - regenerated on every run.

**To reproduce:** put `listingsA2.csv` in `data/`, then run `python EODP_A2_Correlation.py`.
Needs `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

**Still to do:**
- [ ] Convert to notebook cells for `code.ipynb` (after preprocessing, before 2.3)
- [ ] Write 2.2 / 3.2 / 4.2 (Len)
- [ ] Resolve section 5 items with the team
