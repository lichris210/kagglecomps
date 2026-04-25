# Homework 3: Additional Boosting Models

**Chris Li (Kakuzu)**
GSBS 545: Advanced Machine Learning for Business Analytics
Competition: Playground Series S6E4 — Predicting Irrigation Need

---

## Notebook Links

- **LightGBM:** [[Link to Kaggle notebook]](https://github.com/lichris210/kagglecomps/blob/main/notebooks/04_lightgbm.ipynb)
- **CatBoost:** [[Link to Kaggle notebook]](https://github.com/lichris210/kagglecomps/blob/main/notebooks/05_catboost.ipynb)

---

## Modeling Approaches

I trained two additional gradient boosting models — LightGBM and CatBoost — each with three distinct hyperparameter configurations designed to isolate the effect of specific tuning decisions. After comparing the named configs, I ran Optuna hyperparameter tuning (15 trials, stratified CV) on each model, optimizing for balanced accuracy (the competition metric). All models used StratifiedKFold cross-validation to preserve the class distribution across folds, which is critical given the severe class imbalance: High irrigation need represents only 3.3% of the training data.

**What worked well:** Class weighting was by far the most impactful hyperparameter across both models. Adding `class_weight='balanced'` (LightGBM) or `auto_class_weights='Balanced'` (CatBoost) consistently improved balanced accuracy by 0.5–1.0 percentage points. This improvement came almost entirely from better recall on the rare "High" class — for example, in LightGBM, High recall jumped from 0.9147 (no weighting) to 0.9322 (balanced), while Low and Medium recall barely changed.

**What had moderate impact:** L1/L2 regularization and subsampling provided small but consistent gains on top of class weighting. In LightGBM, adding `reg_alpha=5.0`, `reg_lambda=5.0`, and `subsample=0.7` improved balanced accuracy from 0.9664 to 0.9695. In CatBoost, `l2_leaf_reg=10` with Bernoulli subsampling pushed balanced accuracy from 0.9663 to 0.9669. These are meaningful but secondary to class weighting.

**What didn't help much:** Optuna tuning provided diminishing returns. In LightGBM, Optuna's best trial (0.9696) barely improved over Config 3 (0.9695). In CatBoost, Optuna found 0.9672 versus Config 3's 0.9669. This suggests the manual configurations were already near-optimal, and the remaining performance gap lies in feature engineering or ensemble stacking rather than single-model hyperparameter tuning.

**Feature importance validation:** Both models consistently ranked the same features at the top: Rainfall_mm, Soil_Moisture, Temperature_C, Wind_Speed_kmh, Crop_Growth_Stage, and Mulching_Used. These are exactly the 6 features that Chris Deotte (Kaggle Grandmaster) identified in his "Original Data Exact Formula" notebook as fully determining the target in the original source dataset. This confirms the models are learning the real signal rather than fitting noise from the synthetic data generation.

---

## Performance Summary

| Model / Config | CV Bal. Acc. | CV Macro F1 | CV Accuracy | High Recall | Val Bal. Acc. | LB Score |
|---|---|---|---|---|---|---|
| **LGB Config 1: Shallow + Fast** | 0.9614 | 0.9695 | 0.9845 | 0.9147 | — | — |
| LGB Config 2: Deep + Balanced | 0.9664 | 0.9680 | 0.9842 | 0.9322 | — | — |
| LGB Config 3: Regularized + Balanced | 0.9695 | 0.9641 | 0.9833 | 0.9448 | — | — |
| **LGB Optuna Best** | 0.9696 | 0.9598 | 0.9826 | 0.9600 | 0.9723 | 0.96524 |
| **CB Config 1: Shallow + Default** | 0.9574 | 0.9670 | 0.9836 | 0.9042 | — | — |
| CB Config 2: Deep + Balanced | 0.9663 | 0.9528 | 0.9802 | 0.9424 | — | — |
| CB Config 3: Regularized + Balanced | 0.9669 | 0.9518 | 0.9800 | 0.9450 | — | — |
| **CB Optuna Best** | 0.9672 | 0.9520 | 0.9800 | 0.9600 | 0.9703 | 0.96790 |


---

## Model Comparison

LightGBM achieved a higher validation balanced accuracy (0.9723 vs 0.9703), but CatBoost outperformed it on the actual leaderboard (0.96790 vs 0.96524). This reversal highlights a key lesson: local validation scores don't always predict leaderboard performance perfectly, because the test set distribution may differ slightly from the training set. CatBoost's ordered target statistics and symmetric tree structure may generalize better to unseen data despite showing lower CV scores. The models differed primarily in how they handled categorical features: LightGBM treated them as integer-encoded categories and relied more heavily on the numeric features (Rainfall_mm ranked first by a wide margin), while CatBoost's ordered target statistics elevated the two categorical features — Crop_Growth_Stage and Mulching_Used — to 2nd and 4th in its importance rankings.

An interesting trade-off emerged between balanced accuracy and macro F1. LightGBM's Config 3 had a lower macro F1 (0.9641) than Config 1 (0.9695), yet a higher balanced accuracy (0.9695 vs. 0.9614). This is because class weighting improves recall on the rare High class (which balanced accuracy rewards) at the cost of precision (which pulls macro F1 down). Since the competition metric is balanced accuracy, this trade-off is clearly worthwhile.

CatBoost was significantly slower to train than LightGBM, even with GPU acceleration. Its symmetric (oblivious) tree structure produces more stable models but requires more iterations to converge. Interestingly, Optuna selected `depth=4` (the shallowest option) as the best CatBoost configuration, suggesting that CatBoost's built-in regularization from symmetric trees makes deep trees unnecessary for this dataset.

Across all three boosting models (including XGBoost from the previous homework), the same pattern held: the unweighted shallow baseline achieves high accuracy but poor balanced accuracy; adding class weighting provides the biggest single improvement; regularization and subsampling add a smaller but real second-order gain; and Optuna tuning provides marginal improvement on top of well-chosen manual configurations. The consistency of this pattern across XGBoost, LightGBM, and CatBoost suggests it is a property of the dataset and competition metric, not an artifact of any one algorithm.
