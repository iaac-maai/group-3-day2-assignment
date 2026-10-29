---
language: en
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - built-environment
  - iaac-maai
  - ai-methods-1
  - random-forest
  - education
metrics:
  - r2
  - mae
  - rmse
---

# Mumbai monthly rental prediction — Random Forest regression

This model estimates monthly advertised rent in Mumbai, India, from eight property attributes. It is a fitted scikit-learn pipeline containing imputation, categorical one-hot encoding, and a Random Forest regressor. It is intended for students and researchers studying tabular regression, and was developed for **AI Methods I, IAAC MaAI 2026/27**.

**Team:** Ahmed Belahcen, Nandini Rathod, Sanjana Kiran, Akanksha Dhumal. Names are documented in the [Day 1 team repository](https://github.com/iaac-maai/Group3-day1assignment).

**Training source:** [Group 3 Day 2 repository](https://github.com/iaac-maai/group-3-day2-assignment/tree/6b2a6d17af8c4d8950f12e60dc97f8b9a8d4dbc1). This card follows [section 6 of the assignment](https://github.com/iaac-maai/ai-methods-1-2026/blob/main/Day_2/submissions/README.md#6-the-model-card).

## Intended use

Use this model for educational experiments with Mumbai rental listings resembling the training sample. It predicts an advertised price, not an independently assessed market value or an agreed tenancy price.

Out-of-scope uses include other cities or countries, current market pricing without fresh validation, automated housing eligibility or lending decisions, binding valuations, and properties outside the observed data ranges. It has no calibrated uncertainty estimates and is not a production valuation service.

## Model details

| Item | Value |
|---|---|
| Task | Single-output regression; target `Price_num` |
| Published artifact | `model.joblib`, the complete fitted `Pipeline` |
| Pipeline steps | `preprocess` (`ColumnTransformer`), then `model` (`RandomForestRegressor`) |
| Forest parameters | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `random_state=42`, `n_jobs=1` |
| Other key forest settings | `criterion="squared_error"`, `max_features=1.0`, `min_samples_split=2`, `bootstrap=True`, `max_samples=None`, `ccp_alpha=0.0` |
| Numeric processing | Training-set median imputation; **no scaling** |
| Categorical processing | Training-set most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")` |
| Artifact provenance | Byte-for-byte copy of `mumbai_rental_model.joblib` at original Hugging Face revision `97a28ccea3b183a2b3827b279c6741155f6febca` |
| SHA-256 | `b4193b09cc1316008936387fd28f65b07960447feab5d1eaa279ed8745791262` |

The former `mumbai_rental_pipeline.joblib` contains a **LinearRegression** pipeline with numeric scaling. Both original exports are regression models. Neither is a classifier, and neither is the depth-5 Decision Tree whose R² was previously reported as 0.611. The notebook contains those additional educational experiments; this card documents only the fitted Random Forest.

## Input features

Provide a `pandas.DataFrame` with these exact eight column names. Numeric values must already be parsed as numbers; represent missing values with `numpy.nan`. The fitted pipeline does not parse units, extract neighborhoods, or clean strings.

| Column | Type | Unit and meaning | Fitted preprocessing |
|---|---|---|---|
| `Area_num` | float | Reported property area; source declares square feet (ft²) | Median imputation: 1100; no scaling |
| `BHK_num` | float/integer | Bedroom count in the source BHK field | Median imputation: 2; no scaling |
| `Bathrooms_num` | float/integer | Bathroom count | Median imputation: 2; no scaling |
| `Balconies_num` | float/integer | Balcony count; missing means unknown | Median imputation: 1; no scaling |
| `Furnished_clean` | string | `Unfurnished`, `Semi-Furnished`, or `Furnished` | Mode: `Semi-Furnished`; one-hot encoding |
| `Facing` | string | Reported facing direction, retaining source spelling | Mode: `East`; one-hot encoding |
| `Location` | string | City label; **only `Mumbai` occurs in this training dataset** | Mode: `Mumbai`; one-hot encoding |
| `Property Type` | string | Property category, e.g. `Apartment` | Mode: `Apartment`; one-hot encoding |

Unknown categorical values are encoded with all zeros for the corresponding field; this prevents an encoding error but does not establish reliable predictions for unseen categories. `Facing` spellings include inconsistent spacing such as `North - East` and `South -West`; see `evaluation.json` for observed categories.

Observed non-missing ranges: area 250–14,500 reported ft²; BHK 1–5; bathrooms 1–6; balconies 1–5. These are sample ranges, not validated operating guarantees. The [Kaggle description](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset) declares ft², while Day 1 identified possible mixed-unit anomalies and deliberately did not automatically convert them. Supply the source convention; do not silently pass m² values. Per-listing unit correctness remains unverified.

## Output

`predict` returns a one-dimensional NumPy array with one estimated **monthly rent in Indian rupees (INR/₹)** per row. The target is not log-transformed; no exponentiation or inverse target transform is required. Training target values range from ₹1,800 to ₹3,800,000 per month.

## Training data

| Item | Description |
|---|---|
| Dataset | Property Rental Listings Dataset, Harshal H (`harshalhonde`) |
| Source | [Kaggle dataset](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset); scraped from Magicbricks according to the uploader |
| Data licence | **CC BY 4.0**, as declared by the Kaggle API on 2026-10-29; [licence text](https://creativecommons.org/licenses/by/4.0/) |
| Training snapshot | [`data/mumbai_cleaned.csv`](https://github.com/iaac-maai/group-3-day2-assignment/blob/6b2a6d17af8c4d8950f12e60dc97f8b9a8d4dbc1/data/mumbai_cleaned.csv) |
| Snapshot SHA-256 | `354d28e07432137cdc97a2197a556269137e74135af01186420e2e2917a19ee9` |
| One row | One advertised property listing; not necessarily one unique dwelling |
| Size | 18,846 rows, 26 available columns; eight predictors used |
| Split sizes | 15,076 training rows and 3,770 test rows |
| Coverage | Mumbai; listing observation dates are unavailable in the cleaned snapshot |

The Kaggle metadata reports version 2, last updated 2023-08-17. This is an upload timestamp, not a verified collection period, and the exact Kaggle version originally downloaded was not recorded.

**Cleaning:** [Day 1](https://github.com/iaac-maai/Group3-day1assignment) reports 23,405 original records and 18,846 final analytical records. Its notebook documents numeric parsing, filtering invalid prices/areas and implausible counts, area-consistency checks, and preserving unknown balcony values. Day 2 consumes an already cleaned CSV and drops rows missing `Price_num` (none in this snapshot). The entire raw-to-cleaned lineage has not been independently reproduced; the evaluation below is reproducible from the pinned cleaned CSV. Its missing counts are 15 bathrooms, 10,323 balconies, and 13 furnishing values. Imputation and encoding are fitted inside the pipeline on training data only.

## Training procedure

The regression split is `train_test_split(test_size=0.20, random_state=42)`, with default shuffling and **no stratification**, preserving CSV row order before splitting. Classification splits elsewhere in the notebook do not apply here.

The notebook compares Linear Regression, KNN (`n_neighbors=5`), Decision Tree (`max_depth=3`), Random Forest (`n_estimators=200`), and HistGradientBoosting (`min_samples_leaf=3`), with a mean-prediction DummyRegressor baseline. A median baseline is also evaluated in its earlier Decision Tree experiment.

The following comparison means are **saved notebook results**, shown at their recorded precision, using `KFold(n_splits=5, shuffle=True, random_state=42)` on the training set (cell 130). They are not substituted for the audited metrics of the published artifact.

| Candidate | Mean CV R² | Mean CV MAE | Mean CV RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.620 | 42467.323 | 80489.685 |
| HistGradientBoosting | 0.619 | 38517.238 | 80735.054 |
| Decision Tree, depth 3 | 0.615 | 41753.587 | 81208.097 |
| Random Forest, untuned | 0.588 | 37822.608 | 83658.966 |
| KNN, k=5 | 0.575 | 40490.818 | 85287.567 |
| Dummy, mean (audit rerun) | -0.0001691334 | 76826.9055653394 | 131047.6037439728 |

Random Forest tuning uses `GridSearchCV` over `n_estimators=[100,200]`, `max_depth=[10,20,None]`, and `min_samples_leaf=[2,5]`: 12 configurations, five shuffled folds, seed 42, scoring R². The selected parameters are 200 trees, depth 10, and minimum leaf size 5. The tuned forest's mean CV R² is 0.6516348668; the notebook exports `grid_search.best_estimator_` through `best_pipeline`. This explains the artifact choice even though a later plotting cell separately calls a Decision Tree `final_model`.

The selected forest is retained because it is the complete fitted regression artifact actually exported after tuning and has a better recorded tuning CV R² than the untuned candidates. This does not prove it is universally optimal. CV is not nested: the same folds informed hyperparameter selection, so the selected CV score is optimistic as an estimate of an independent selection procedure. Several notebook experiments also inspect the same test set; the reported test results should not be treated as a pristine final benchmark.

## Evaluation

The following metrics were recomputed on 2026-10-29 from the pinned CSV and the original exported Random Forest. A fresh fit with its parameters reproduced **all test predictions within absolute tolerance 1e-8**. Numbers below are rounded for readability; `evaluation.json` stores full floating-point values and each CV fold.

### Held-out test set (3,770 rows)

| Model | R² | MAE (INR/month) | RMSE (INR/month) |
|---|---:|---:|---:|
| **Published Random Forest** | **0.6187457378** | **38044.9279004604** | **82899.4548739774** |
| Dummy, training mean | -0.0000000116 | 77182.5284172923 | 134259.2829850732 |
| Dummy, training median | -0.0954835938 | 67123.7538461538 | 140522.9504965741 |

### Five-fold CV on the training set: mean ± standard deviation

All rows use `KFold(n_splits=5, shuffle=True, random_state=42)` and freshly fitted estimators per fold. Standard deviation uses `ddof=0` across the five validation scores, and is not a confidence interval. Error scoring signs are converted to positive MAE/RMSE.

| Model | R² | MAE (INR/month) | RMSE (INR/month) |
|---|---:|---:|---:|
| **Published forest configuration** | **0.6516348668 ± 0.0704462869** | **37250.7823349273 ± 711.5192966437** | **77140.4309250284 ± 10150.5695328676** |
| Dummy, mean | -0.0001691334 ± 0.0001406681 | 76826.9055653394 ± 869.3534260931 | 131047.6037439728 ± 6418.6541357984 |
| Dummy, median | -0.1006516249 ± 0.0063253254 | 67274.0457842969 ± 1384.1494587204 | 137457.7048402592 ± 6435.0418205557 |

The earlier Decision Tree test scores (MAE 40528.67, RMSE 83764.35, R² 0.611) and its unshuffled CV results describe another experiment and must not be attributed to `model.joblib`.

**Where it performs worst:** test MAE is ₹19,770.47 for rents ≤₹100,000 (2,597 rows), ₹78,504.23 above ₹100,000 (1,173 rows), and ₹295,559.46 above ₹500,000 (68 rows; a subset of the preceding band). These are descriptive audit slices, not additional model selection. Neighborhood performance cannot be measured from `Location`, which is constant.

## Limitations and biases

- Online advertised listings are a selective sample, not a representative survey of households or completed rentals. Luxury outliers produce large errors.
- There are 2,487 repeated rows when considering the eight predictors plus target. In the random split, 752 test rows match at least one training row on those fields. These matches do not prove identical dwellings, but create a substantial dependence/leakage concern. Group-aware deduplication and a new independent split should precede deployment.
- `Location` contains only `Mumbai`. The model does not use neighborhoods inferred from titles elsewhere in Day 1. Claims about learning neighborhood price differences would be unsupported.
- Balcony missingness is high; median imputation does not recover the underlying information. Mixed area units, inconsistent labels, and unverified cleaning lineage remain risks.
- `Price`, `Price_num`, `Price_per_m2`, and `Suspicious_Price` are excluded from predictors. This prevents their direct target leakage, but does not certify the upstream cleaning as leakage-free.
- No temporal validation, external city validation, fairness audit, or uncertainty calibration has been performed. Historical scores do not establish accuracy for today's market.

## How to use

Install `requirements.txt` in a virtual environment. Only load this artifact if you trust its publisher. After the correction is merged, download `model.joblib` from `main`; for a Pull Request preview, set `REVISION` to `refs/pr/<number>`. For long-term reproducibility, use the reviewed full commit SHA.

```python
import hashlib
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REVISION = "main"  # Replace with the reviewed commit SHA after merge.
path = hf_hub_download(
    repo_id="iaac-maai/group3-day2-mumbairental",
    filename="model.joblib",
    revision=REVISION,
)
expected = "b4193b09cc1316008936387fd28f65b07960447feab5d1eaa279ed8745791262"
with open(path, "rb") as file:
    assert hashlib.sha256(file.read()).hexdigest() == expected
model = joblib.load(path)  # Pickle-based: trusted artifacts only.
example = pd.DataFrame([{
    "Area_num": 330.0,
    "BHK_num": 1.0,
    "Bathrooms_num": 1.0,
    "Balconies_num": np.nan,
    "Furnished_clean": "Furnished",
    "Facing": "East",
    "Location": "Mumbai",
    "Property Type": "Apartment",
}])
print(f"Predicted monthly rent: INR {model.predict(example)[0]:,.2f}")
# Predicted monthly rent: INR 25,586.09
```

The example is the first row of the reconstructed test split; its observed monthly rent is INR 30,000. A standalone equivalent is provided as `predict_example.py`, supporting `--model model.joblib` for local verification and `--revision` for a pinned Hub download.

## Reproducibility

The preserved `supervised_learning.ipynb` remains the original record of all experiments. The published artifact was created with scikit-learn **1.9.1**, pandas **3.1.0**, NumPy **2.5.4**, SciPy **1.18.1**, joblib **1.6.0**, and Python **3.14.7**, according to notebook outputs. The audit and inference also succeeded under Python **3.12.14**, with those same five package versions. See `evaluation.json` for the exact audit runtime and `requirements.txt` for inference dependencies.

Run `python reproduce_evaluation.py --data path/to/mumbai_cleaned.csv --model model.joblib` to recompute artifact test scores and five-fold CV for its cloned configuration and the two baselines. The script verifies both file hashes before loading. It does not rerun the entire exploratory notebook or the 12-configuration search. The original GitHub `requirements.txt` is unpinned; the inference requirements here are pinned to the successfully tested runtime.

## Security note

**Joblib uses pickle-based serialization and loading a malicious file can execute arbitrary code.** Load files only from a publisher and revision you trust, ideally in an isolated environment without secrets. The supplied SHA-256 detects changes relative to this reviewed artifact; a checksum alone does not prove safety. Reuse the tested scikit-learn version: cross-version pickle compatibility is not guaranteed.

## Authors, course and licence

**Ahmed Belahcen, Nandini Rathod, Sanjana Kiran, Akanksha Dhumal — Group 3, AI Methods I, IAAC MaAI 2026/27.**

**Model and code: MIT**, following the project owner's explicit instruction and the existing [source repository MIT licence](https://github.com/iaac-maai/group-3-day2-assignment/blob/6b2a6d17af8c4d8950f12e60dc97f8b9a8d4dbc1/LICENSE), copyright (c) 2026 IAAC - MAAI. The licence text is included as `LICENSE`. Organization maintainers must approve publication and merge; this card does not independently establish every contributor's authority to license organizational assets.

**Training data: CC BY 4.0**, separately declared by Kaggle; credit Harshal H (`harshalhonde`), link the source and licence, and identify the team's cleaning modifications. The model/code MIT declaration does not relicense the dataset or establish rights to the original Magicbricks material beyond the uploader's declaration. No raw listing descriptions or images are needed for inference or added by this correction.
