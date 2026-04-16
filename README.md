# Kaggle Playground Series - S6E4: Predicting Irrigation Need

## Competition
[Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4) — Classification task predicting irrigation need based on environmental and soil features.

## Repository Structure

```
kagglecomps/
├── data/                  # Competition data (not tracked by git)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   ├── 02_bagging.ipynb   # Bagging models (Random Forest, etc.)
│   └── 03_boosting.ipynb  # Boosting models (XGBoost, LightGBM, CatBoost)
├── submissions/           # Kaggle submission CSVs
├── kaggle_homework.md     # Homework document with URLs, discussion, metrics
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API key)
kaggle competitions download -c playground-series-s6e4 -p data/
unzip data/playground-series-s6e4.zip -d data/
```

## Kaggle API Setup
1. Go to https://www.kaggle.com → Your Profile → Settings → API → Create New Token
2. Save `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`
