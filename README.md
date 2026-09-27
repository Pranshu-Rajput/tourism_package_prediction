#  Wellness Tourism Package — Purchase Prediction (MLOps CI/CD)

An end-to-end MLOps pipeline that predicts whether a customer will buy the **Wellness Tourism Package** from **"Visit with Us"** *before* a salesperson contacts them. Data registration, preparation, model training, experiment tracking, and deployment are all automated with **GitHub Actions**, and the trained model is served through a **Streamlit** web app.

🔗 **Live app:** https://tourismpackageprediction-48dqwmthfuxtxsnaz5mdsb.streamlit.app/

---

##  Business Problem

"Visit with Us" currently identifies potential customers manually. The process is inconsistent, slow, and error-prone, so sales calls go to the wrong people and campaigns underperform. This project replaces it with a scalable, automated system that:

- predicts likely buyers from customer and sales-interaction data,
- retrains and redeploys automatically whenever the code or data changes,
- gives the sales team a simple app for scoring new customers.

---

##  Pipeline Architecture

```
push to main
     │
     ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
│ 1. Register      │──▶│ 2. Data          │──▶│ 3. Model Training    │──▶│ 4. Deployment    │
│    Dataset       │   │    Preparation   │   │    + MLflow Tracking │   │    Prep          │
│ validate schema  │   │ clean, impute,   │   │ GridSearchCV (180),  │   │ smoke-test the   │
│ print summary    │   │ stratified split │   │ log all trials, save │   │ committed model  │
└──────────────────┘   └──────────────────┘   │ & commit model       │   └──────────────────┘
                                              └──────────────────────┘            │
                                                                                  ▼
                                                                  Streamlit Community Cloud
                                                                  (auto-redeploys from main)
```

- **Trigger:** every push to `main`, or a manual run from the Actions tab.
- **Artifacts:** jobs pass the registered data, the train/test splits, and the trained model to each other.
- **Model commit:** the pipeline commits the model back to the repo with `[skip ci]`, so the commit doesn't start another run.
- **Concurrency:** if a newer push arrives, it cancels any older run that's still in progress.

---

##  Repository Structure

```
tourism_package_prediction/
├── .github/
│   └── workflows/
│       └── pipeline.yml                   # CI/CD workflow (4 jobs)
├── tourism_project/
│   ├── requirements.txt                   # pipeline dependencies
│   ├── data/
│   │   └── tourism.csv                    # registered dataset
│   ├── model_building/
│   │   ├── data_register.py               # Job 1 – schema validation & summary
│   │   ├── prep.py                        # Job 2 – cleaning, imputation, split
│   │   └── train.py                       # Job 3 – tuning, tracking, evaluation, export
│   └── deployment/
│       ├── app.py                         # Streamlit app
│       ├── requirements.txt               # app dependencies
│       └── best_tourism_model_v1.joblib   # committed by the pipeline
└── README.md
```

---

##  Dataset

The dataset has **4,128 records and 20 attributes** describing customer profiles and sales interactions. The target is `ProdTaken` (1 = purchased).

| Group | Features |
|---|---|
| Customer details | `Age`, `TypeofContact`, `CityTier`, `Occupation`, `Gender`, `NumberOfPersonVisiting`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `OwnCar`, `NumberOfChildrenVisiting`, `Designation`, `MonthlyIncome` |
| Sales interaction | `PitchSatisfactionScore`, `ProductPitched`, `NumberOfFollowups`, `DurationOfPitch` |

**Class balance:** 80.7% non-buyers and 19.3% buyers. Because of this imbalance, the model is tuned on **F1**, not accuracy.

---

##  Data Preparation

- Dropped the identifier columns `CustomerID` and `Unnamed: 0`.
- Standardised the `Gender` label (`"Fe Male"` → `"Female"`).
- Removed **117 duplicate records**, leaving 4,011 unique rows.
- Applied a **stratified 80/20 split**: 3,208 training rows and 803 test rows.

**Feature treatment**

| Group | Columns | Imputation | Encoding |
|---|---|---|---|
| Continuous | `Age`, `DurationOfPitch`, `NumberOfTrips`, `MonthlyIncome` | Median | Standard scaling |
| Discrete numeric (treated as categorical) | `CityTier`, `NumberOfPersonVisiting`, `NumberOfFollowups`, `PreferredPropertyStar`, `Passport`, `PitchSatisfactionScore`, `OwnCar`, `NumberOfChildrenVisiting` | Mode | One-hot |
| Nominal | `TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation` | Mode | One-hot |

Preprocessing lives inside the scikit-learn `Pipeline`, so it is fitted separately within each CV fold (no leakage) and ships with the model.

---

##  Model & Results

**Model:** Decision Tree Classifier, tuned with `GridSearchCV`. The search covered 180 combinations with 5-fold CV and F1 scoring, and **every trial was logged to MLflow**.

**Best parameters:** `criterion=entropy`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight=None`

| Metric | Test score |
|---|---|
| Accuracy | **0.908** |
| Precision | **0.783** |
| Recall | **0.723** |
| F1 | **0.752** |
| ROC-AUC | **0.837** |
| Best CV F1 | 0.688 |

**Confusion matrix (test, n = 803)**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 617 | 31 |
| **Actual Yes** | 43 | 112 |

> About 78% of customers the model flags actually buy, compared with a 19.3% base rate. That makes outreach lists roughly **4× richer in buyers** than random calling.

---

##  Business Recommendations

1. **Prioritise model-flagged customers** for Wellness package calls, and send low-probability customers to lower-cost channels.
2. **Design segment-specific campaigns** around the strongest drivers the model identifies.
3. **Standardise the sales pitch** by using the follow-up and pitch patterns linked to higher conversion.
4. **Tune the decision threshold** to match capacity: favour recall during launches and precision when sales capacity is limited.
5. **Retrain regularly** through the pipeline as new campaign outcomes come in.

---

##  How to Run

### Reproduce locally
```bash
git clone https://github.com/Pranshu-Rajput/tourism_package_prediction.git
cd tourism_package_prediction
pip install -r tourism_project/requirements.txt

python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
mlflow ui --port 5000 &                      # tracking server
python tourism_project/model_building/train.py
```

### Run the app locally
```bash
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

### Trigger the pipeline
Push to `main`, or go to **Actions → Tourism Package Prediction CI/CD Pipeline → Run workflow**.

---

##  Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, pandas, NumPy |
| Experiment tracking | MLflow |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |
| Automation | PyGithub |

---

## 👤 Author

**Pranshu Rajput** · [GitHub](https://github.com/Pranshu-Rajput)
