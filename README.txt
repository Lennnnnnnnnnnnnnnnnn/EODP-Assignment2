COMP20008 ASSIGNMENT 2 - CODE README
Group W05G05

Research question:
  Is the pricing of a property linked to its overall rating, and what kinds of
  variation in pricing exists between properties?


-------------------------------------------------------------------------------
1. WHAT IS SUBMITTED
-------------------------------------------------------------------------------

  code.ipynb    All code for the report, in one notebook, in report order.
  README.txt    This file.

The notebook is also kept in sync with five standalone scripts used during
development (EODP_A2_Preprocessing.py, EODP_A2_Correlation.py,
EODP_A2_Supervised.py, EODP_A2_FeatureSelection.py, EODP_A2_PCA&Clustering.py).
The notebook is the submitted artefact; the scripts are not required to run it.


-------------------------------------------------------------------------------
2. HOW TO RUN
-------------------------------------------------------------------------------

Step 1. Get the dataset.

  The dataset is not submitted with this notebook. Download listings.csv for
  Melbourne, Victoria, Australia from Inside Airbnb:

      http://insideairbnb.com/get-the-data/

Step 2. Put it in the right place.

  Rename the file to listingsA2.csv and place it in a data/ folder beside the
  notebook, so the layout is:

      code.ipynb
      README.txt
      data/
          listingsA2.csv

  The notebook loads it with a relative path (data/listingsA2.csv). Do not use
  an absolute path, as it will not run on another machine.

Step 3. Install the packages.

      pip install pandas numpy scipy scikit-learn matplotlib seaborn

  Developed on Python 3.13 with pandas 2.x, scikit-learn 1.9, matplotlib 3.10.

Step 4. Run it.

  Open code.ipynb and choose Run All. Cells must run in order, top to bottom,
  because every later section uses the dataframes built in section 2.

  Expect roughly five minutes. The slowest cells are the KNN hyperparameter
  sweep in section 4 and the Ward linkage in section 6.

  A figures/ folder is created automatically. Every figure is both displayed
  inline and written there as a 200 dpi PNG.

Reproducibility:
  All random operations are seeded (random_state=42 in section 4, 3000 in
  section 6), so every number and figure below reproduces exactly on a re-run.


-------------------------------------------------------------------------------
3. WHAT THE IMPLEMENTATION DOES
-------------------------------------------------------------------------------

Section 2, Preprocessing (Sabir)
  Parses price from text to numeric. Diagnoses the missingness in
  review_scores_rating as MNAR and keeps only genuinely rated listings rather
  than imputing. Computes haversine distance from the Melbourne CBD and bands
  it into inner/middle/outer. Consolidates 82 raw property_type values into 5
  dwelling groups. Produces df (all listings) and df_rating (rated subset),
  which every later section builds on.

Section 3, Correlation analysis (Len)
  Computes Pearson, Spearman, Mutual Information and Normalised Mutual
  Information for all 21 pairs in a 7-variable set. Pearson and Spearman use
  log(price); MI and NMI use equal-frequency bins. Pairwise deletion is used,
  so n is reported per pair. Pairs involving a nominal variable report n/a for
  Pearson and Spearman, since those methods need ordered values.

Section 4, Supervised learning (Ten)
  Discretises price into three equal-count bands (the classification target),
  builds a 13-column feature matrix, and splits 80/20 with stratification.
  Compares three feature-set configurations, then sweeps 11 values of
  n_neighbors across two weighting schemes and 12 tree depths, using 5-fold
  stratified cross-validation on the training set only. Refits both models at
  the chosen settings and scores them once on the held-out test set, with
  per-class metrics, confusion matrices and a 2000-sample bootstrap confidence
  interval.

Section 5, Feature selection (Ben)
  A filter method (mutual information against continuous price) and an embedded
  method (the tuned decision tree from section 4), with the top 3 features from
  each and a hard-case listing identified by id.

Section 6, PCA and clustering (Ben)
  Chooses k using a VAT heatmap and the elbow method, runs K-Means and Ward
  hierarchical clustering at the same k, and compares the two by their mean
  cluster profiles and side by side in PCA space. Applies PCA to the same
  feature set for the component loadings and two 2D projections, one coloured
  by cluster and one by price.


-------------------------------------------------------------------------------
4. HOW OUTPUTS MAP TO THE REPORT
-------------------------------------------------------------------------------

Notebook section 2  ->  report 2.1 / 3.1 / 4.1 / 5.1

  "null ratings: 4477 (17.4%)" and "all nulls are 0-review: True"
      The MNAR diagnosis and the before/after row counts in 3.1.
  "Spearman price~rating | drop 0.0921 | impute 0.0796"
      The measured impact of the drop-versus-impute choice in 3.1 and 4.1.
  cbd_band and property_group summary tables
      The band sizes and median prices quoted in 3.1.
  "BEFORE 82 categories ... AFTER 5 groups"
      The category-collapse figures in 3.1.

Notebook section 3  ->  report 2.2 / 3.2 / 4.2 / 5.2

  "All 21 pairs" table
      The full correlation table in 3.2. Column n is the per-pair sample size.
  "Pairs with price, sorted by NMI"
      The ranking of predictors against the target discussed in 4.2.
  figures/correlation_heatmaps.png
      The Spearman and NMI heatmap figure in 3.2.

Notebook section 4  ->  report 2.3 / 3.3 / 4.3 / 5.3

  "cut points ($)" and the class balance block
      The price band definitions ($194.25 and $307.00) and the 33.3/33.4/33.3
      split quoted in 2.3 and 3.3. Modelling set is 16,375 listings.
  "majority-class (0R) baseline accuracy: 0.3337"
      The baseline every model is compared against in 3.3.
  "FEATURE SET CHOICE" table
      The evidence for keeping both accommodates and bedrooms, in 2.3.
  "HYPERPARAMETER SWEEP" tables
      The score at every hyperparameter value tried, required in 3.3.
      Also plotted as figures 1 and 2.
  "hyperparameter justification" block
      The default / chosen / effect statement required in 2.3.
  "headline test-set metrics"
      KNN 0.6568 accuracy and Decision Tree 0.6605, with the absolute
      improvement over baseline, in 3.3.
  "per-class precision / recall / F1"
      The per-class table in 3.3.
  "confusion matrices"
      Printed, and plotted as figure 3.
  "UNCERTAINTY" block
      The bootstrap 95% confidence intervals in 3.3. The intervals overlap, so
      the two models are not separable on performance, as discussed in 4.3.
  "FEATURE INFLUENCE" table
      The Gini importances in 3.3, plotted as figure 4, and the basis for the
      rating conclusion in 4.3.

Notebook section 5  ->  report 2.4 / 3.4 / 4.4 / 5.4

  "Top 3 Filter Features (Mutual Information)"
      The filter method's top 3 in 3.4.
  "Top 3 Embedded Features (Decision Tree)"
      The embedded method's top 3 in 3.4. These come from the tuned depth-8
      tree fitted in section 4, so they match the importances reported in 3.3.
  The single printed listing row
      The hard case identified by id in 3.4 and discussed in 4.4.

Notebook section 6  ->  report 2.5 / 3.5 / 4.5 / 5.5

  figures/VAT_Heatmap.png and figures/Elbow_Method.png
      The justification for k in 2.5.
  "PCA Loadings" and the explained variance lines
      The component loadings and explained variance ratios in 3.5.
  figures/K-Means.png and figures/PCA_with_price.png
      The 2D PCA projections, coloured by cluster and by price, in 3.5.
  figures/Dendogram.png and figures/K-means_vs_Hierarchical.png
      The hierarchical clustering figures in 3.5.
  "Average Property Profile per Cluster" and the hierarchical equivalent
      The cluster descriptions in 3.5, and the basis for comparing the two
      methods' groupings in 4.5.


-------------------------------------------------------------------------------
5. NOTES ON CONSISTENCY BETWEEN SECTIONS
-------------------------------------------------------------------------------

Sections 3, 4 and 5 all start from the same base: rated listings that also have
a price, which is 16,375 of the 25,728 listings in the raw file.

Two deliberate differences are worth knowing when reading numbers across
sections:

  bedrooms. Section 3 uses pairwise deletion, so its bedrooms pairs have
  n = 14,058. Sections 4 and 5 impute bedrooms with the median within each
  room_type, keeping all rows, because the missingness is structural (57.1% of
  private rooms have no bedroom count against 2.7% of entire homes).

  Price outliers. Section 5's filter method and all of section 6 exclude
  listings at or above $1,000, which removes 273 rows. Sections 3 and 4 keep
  them. The embedded method in section 5 uses the section 4 tree, so it is
  computed on the full 16,375.

The correlation between accommodates and bedrooms is reported more than once.
Use the section 3 figure, Spearman 0.8643 at n = 14,058, as the canonical one.


-------------------------------------------------------------------------------
6. GENERATIVE AI DECLARATION
-------------------------------------------------------------------------------

Generative AI tools (Claude, ChatGPT, Gemini) were used throughout the project
to assist with writing and debugging code, to explain library behaviour, and to
help refine the wording of drafts we had written ourselves. Where a particular
block leaned on it heavily this is also noted inline in the source, for example
the hierarchical clustering comparison in section 6.

All design and analytical decisions, including variable selection, model choice,
parameter values and the interpretation of results, were made by the group
members, and each member can explain the reasoning behind their own section.


-------------------------------------------------------------------------------
7. DATA SOURCE
-------------------------------------------------------------------------------

Inside Airbnb. (2026). Melbourne, Victoria, Australia.
Retrieved from http://insideairbnb.com/get-the-data/
