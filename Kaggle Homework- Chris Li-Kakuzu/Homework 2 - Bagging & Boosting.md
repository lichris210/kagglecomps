# Kaggle Homework - Chris Li - Kakuzu

## Competition
**Playground Series S6E4: Predicting Irrigation Need**  
[Competition Link](https://www.kaggle.com/competitions/playground-series-s6e4)

---

## 1. Upvoted Notebooks & Discussions

| # | Title | URL | Why I Chose This Notebook |
|---|-------|-----|---------------------------|
| 1 | **Original Data Exact Formula** (by cdeotte) | [Link](https://www.kaggle.com/code/cdeotte/original-data-exact-formula) | Chris Deotte (Kaggle Grandmaster) reverse-engineered the exact formula behind the original dataset. He proved the target is fully determined by just 6 features — four binary thresholds (`Soil_Moisture < 25`, `Temperature_C > 30`, `Rainfall_mm < 300`, `Wind_Speed_kmh > 10`) plus two categoricals (`Crop_Growth_Stage`, `Mulching_Used`) — achieving **100% balanced accuracy** with a simple logistic regression. This insight reveals that the other 13 columns are noise and should guide all downstream feature engineering. |
| 2 | **S6E4: 0.970 · Stacked LGB+XGB+CAT · Feature Engine** (by aliafzal9323) | [Link](https://www.kaggle.com/code/aliafzal9323/s6e4-0-970-stacked-lgb-xgb-cat-feature-engine) | Best overall pipeline architecture. Blends the original source dataset with competition data (a key Playground Series technique), builds domain-driven interaction features (Evaporative Stress, Water Balance, Moisture Deficit, Dryness Index), trains a 3-model GPU ensemble (LightGBM + XGBoost + CatBoost), and stacks with a logistic regression meta-learner. Also includes post-prediction threshold optimization for the rare "High" class. Good template for a competitive end-to-end solution. |
| 3 | **PSS6E4 LGB Baseline (CV 0.97943)** (by yunsuxiaozi) | [Link](https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943) | Highest single-model CV score of the four. Uses digit-extraction feature engineering to detect synthetic data rounding artifacts, target encoding for categoricals, balanced sample weights, and Optuna-based post-hoc class weight optimization on OOF predictions. Demonstrates how to squeeze maximum performance from a single LightGBM model. Worth noting: uses `KFold` instead of `StratifiedKFold`, which is risky given the severe class imbalance (3.3% "High" class). |
| 4 | **S6E4 Irrigation Prediction — LightGBM Baseline** (by sarcasmos) | [Link](https://www.kaggle.com/code/sarcasmos/s6e4-irrigation-prediction-lightgbm-baseline) | Cleanest and most well-documented notebook of the four. Extensive EDA with professional visualizations (target distribution, feature histograms, correlation heatmaps, boxplots, confusion matrices). Uses `StratifiedKFold` (the correct choice for imbalanced data) and `class_weight='balanced'`. Good reference for how to present analysis clearly, though the modeling is simpler (no feature engineering, no original data blending) — CV balanced accuracy ~0.966. |

---

## 2. My Notebook Links

- **EDA Notebook:** [link to notebooks/01_eda.ipynb or Kaggle link]
- **Bagging Notebook:** [link to notebooks/02_bagging.ipynb or Kaggle link]
- **Boosting Notebook:** [link to notebooks/03_boosting.ipynb or Kaggle link]

---

## 3. EDA Insights

- **Key features:** Soil_Moisture is the strongest numeric predictor (correlation of -0.25 with target). The mean for High irrigation need is 17.7 vs 43.3 for Low, which makes sense since drier soil needs more water. For categoricals, Crop_Growth_Stage stands out the most. Flowering and Vegetative stages have about 6% High and 62% Medium need, while Sowing and Harvest have almost no High need and 85%+ Low. Mulching_Used is also a strong signal: with mulching, High need drops from 5.9% to 0.8% and Low jumps from 44.5% to 73%. Temperature, wind speed, and rainfall are secondary but useful predictors. A lot of features turned out to be weak, including Organic_Carbon, Sunlight_Hours, Previous_Irrigation_mm, and Field_Area_hectare, all with near-zero correlations and basically identical means across target classes. Most other categoricals like Soil_Type, Season, and Region only showed 1-3 percentage point differences across classes.
- **Potential issues:**
  - **Class imbalance:** Low makes up 58.7% of the target, Medium 37.9%, and High only 3.3%. A model that just predicts Low every time gets 59% accuracy for free. I'll need to use stratified cross-validation and class weighting to make sure the model doesn't just ignore the High class. Macro F1 is probably a better metric than accuracy here.
  - **No missing data**, so no imputation needed across all 630k rows.
  - **Categorical encoding:** 8 string features need encoding. CatBoost can handle them directly, but for RF and XGBoost I'll need label or one-hot encoding. Most have 6 or fewer levels so one-hot won't blow up the feature space.
  - Some features are probably just noise and could hurt performance. Feature selection or regularization might help.
- **Anything new tried:** Instead of just looking at correlations, I grouped numeric features by target class and compared means/standard deviations, and used crosstab proportions for categoricals. This ended up being way more useful than correlation alone for finding the important features, especially for categoricals like Crop_Growth_Stage and Mulching_Used where correlation doesn't apply directly.

---

## 4. Modeling Approaches

### Bagging
- Model(s) used: Random Forest (scikit-learn RandomForestClassifier)
- Cross-validation strategy: 5-fold Stratified K-Fold, evaluated on both macro F1 and accuracy. Used a 80/20 stratified train/val split for final evaluation after tuning.
- Tuning done: Ran Optuna with 10 trials on a 150k subsample to keep runtime reasonable, then refit best params on the full 630k training set. Tuned n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, and class_weight. Also compared unweighted vs balanced class weights as separate baselines before tuning. Best params: n_estimators=221, max_depth=20, min_samples_split=14, min_samples_leaf=5, max_features=None, class_weight=None.

### Boosting
- Model(s) used: XGBoost (XGBClassifier with GPU acceleration on Kaggle)
- Cross-validation strategy: Same 5-fold Stratified K-Fold setup, same 80/20 val split. Used manual CV loop instead of cross_val_score for the weighted baseline so sample_weight could be passed during fit.
- Tuning done: Ran Optuna with 50 trials on the full training set using Kaggle's T4 GPU. Tuned n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_lambda, and whether to use balanced sample weights. Best params: n_estimators=573, max_depth=3, learning_rate=0.25, subsample=0.70, colsample_bytree=0.88, min_child_weight=1, gamma=1.27, reg_lambda=0.97, use_weights=False.

### Performance Comparison

| Model | CV Macro F1 | Val Macro F1 | Val Accuracy | Kaggle LB | Notes |
|-------|-------------|--------------|--------------|-----------|-------|
| RF baseline (no weight) | 0.9692 | — | — | — | Default params, 300 trees |
| RF balanced weights | 0.9683 | — | — | — | Weighting hurt slightly |
| RF Optuna-tuned | 0.9696 (150k sample) | 0.9721 | 0.9861 | 0.95981 | Best bagging model |
| XGB baseline (no weight) | 0.9696 | — | — | — | Default params, 300 trees, GPU |
| XGB balanced weights | 0.9703 | — | — | — | Slight F1 improvement |
| XGB Optuna-tuned | 0.9701 | 0.9717 | 0.9853 | 0.96053 | 50 trials, best boosting model |

Class-level breakdown (both tuned models, on validation set):
- High: 96% precision, 92% recall (hardest class, only 3.3% of data)
- Low: 99% precision, 100% recall
- Medium: 99% precision, 98% recall

---

## 5. Boosting vs. Bagging Discussion

RF slightly beat XGBoost on local validation (F1 0.9721 vs 0.9717), but on the Kaggle leaderboard the result flipped: XGBoost scored 0.96053 vs RF at 0.95981. This suggests the RF model may have slightly overfit to the training data, while XGBoost generalized better to unseen test data, possibly due to its built-in regularization (gamma, reg_lambda). This is more in line with what you'd typically expect — boosting usually edges out bagging on structured tabular data.

Both models hit the same ceiling and struggle in the same spot: the "High" class (92% recall). Class weighting didn't help either model. The remaining gains are probably in feature engineering and ensembling, not model selection.

---

## 6. Phase 2 Plan

- Feature engineering: interaction features (Soil_Moisture × Temperature, Rainfall / Field_Area), target encoding for categoricals, binary Flowering/Vegetative vs Sowing/Harvest feature
- Ensemble stacking: combine XGBoost, LightGBM, CatBoost, and RF predictions as inputs to a meta-model
- Re-tune XGBoost with wider max_depth range, also try LightGBM and CatBoost
- Seed averaging across 5-10 random seeds for free gains
- Set up an automated experiment pipeline: use a script that iterates through model configs, logs results to a structured file, and run it continuously on Kaggle notebooks or a cloud GPU (Colab Pro / Lambda Labs). Use Claude Code to read the results log, analyze what's working, and generate the next batch of experiments. This lets me run 50+ experiments overnight instead of 5 manually.
- Review top public notebooks for new techniques