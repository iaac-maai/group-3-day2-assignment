# Mumbai rental learning experiments and published regression model

## Published model and reproducibility

The single published result is a **tuned Random Forest regression pipeline** estimating monthly Mumbai rent in INR from eight property attributes. The notebook also retains Decision Tree regression and furnishing classification experiments for teaching; those are not the published artifact.

- **Model Card:** [models/README.md](models/README.md), following section 6 of the assignment.
- **Hugging Face model:** https://huggingface.co/iaac-maai/group3-day2-mumbairental
- **Reviewed correction:** https://huggingface.co/iaac-maai/group3-day2-mumbairental/discussions/4 (available through `refs/pr/4` until merge).
- **Notebook:** [notebooks/supervised_learning.ipynb](notebooks/supervised_learning.ipynb); see its final **Final published model** section, which runs independently of the earlier classifier variables.
- **Evaluation:** [models/evaluation.json](models/evaluation.json) stores full-precision test scores, CV folds, baselines and provenance.

The forest uses 200 trees, maximum depth 10, minimum leaf size 5, and seed 42. The regression split is 15,076 train / 3,770 test rows, shuffled 80/20, seed 42, without stratification. Numeric median imputation has no scaling; categorical mode imputation precedes one-hot encoding.

| Published forest metric | Test | Five-fold training CV mean ± std |
|---|---:|---:|
| R² | 0.6187457378 | 0.6516348668 ± 0.0704462869 |
| MAE (INR/month) | 38044.9279004604 | 37250.7823349273 ± 711.5192966437 |
| RMSE (INR/month) | 82899.4548739774 | 77140.4309250284 ± 10150.5695328676 |

CV uses five shuffled KFold splits with seed 42 and `ddof=0`. It is not nested and was used in tuning. `Location` is constant (`Mumbai`); 752 test rows match training rows on the predictors plus target. Historical listing data, repeated rows and possible mixed area units limit generalization. The Model Card explains these issues and includes mean/median baseline results. The older Decision Tree R² of 0.611 describes a different model.

### Install and test

From the repository root, use Python 3.12 (audit tested on 3.12.14) and a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python models/predict_example.py --revision refs/pr/4
```

The download example verifies the reviewed artifact hash and prints **INR 25,586.09/month**. After the Hugging Face PR is merged, use its reviewed full commit SHA in place of `refs/pr/4`. The original notebook reports Python 3.14.7; its ML package versions are pinned here. Matplotlib and seaborn pins come from the saved notebook environment; the appended regression section was tested separately without rerunning every historical experiment.

To verify the published artifact's metrics, download its trusted `model.joblib` into `models/`, then run:

```bash
python models/reproduce_evaluation.py --data data/mumbai_cleaned.csv --model models/model.joblib
```

Alternatively, run the notebook's final section in a fresh kernel to train the selected configuration, calculate test/CV metrics and export/reload `models/model.joblib`. It refuses to overwrite an existing export. A fresh serialization can have a different hash despite matching predictions; use the original Hub file for the hash-pinned script above. Generated joblib files are ignored; historical exports already tracked under `notebooks/models/` remain for traceability.

**Security:** joblib/pickle loading can execute code; only load trusted files and revisions. Checksums identify the reviewed bytes but do not prove safety.

**Authors:** Ahmed Belahcen, Nandini Rathod, Sanjana Kiran, Akanksha Dhumal. **Course:** AI Methods I, IAAC MaAI 2026/27. **Model/code:** MIT, see [LICENSE](LICENSE). **Training data:** Harshal H (`harshalhonde`), [Kaggle Property Rental Listings Dataset](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset), separately declared CC BY 4.0; the team's cleaned snapshot is a modification, not an MIT relicensing of the data.

## Preserved exploratory classification report

The original classification report below is retained as the historical teaching experiment. Its ROC-AUC of approximately 0.577 belongs to the earlier unrestricted classifier and is not a validated score for the later depth-5 classifier or the published forest.

# Mumbai Property Rental Classification Using Machine Learning

## 1. Project Overview

This project applies supervised machine learning to a dataset of property rental listings from Mumbai, India. The objective is to predict a property's furnishing type using information such as its area, number of bedrooms, bathrooms, balconies, facing direction, location, and property type.

The project explores how data cleaning, preprocessing, and classification models can be used to investigate patterns in real estate data.

## 2. Objectives

- Clean and prepare a real-world property rental dataset.
- Explore the dataset and understand its features.
- Apply preprocessing techniques to numerical and categorical variables.
- Train a supervised machine learning classification model.
- Evaluate model performance using multiple classification metrics.
- Identify which property features contribute most to predictions.

## 3. Dataset

**Dataset:** Property Rental Listings Dataset  
**Geographical focus:** Mumbai, India  
**Source:** [Kaggle — Property Rental Listings Dataset](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset)

The dataset contains property listing information, including:

- Property area
- Number of bedrooms (BHK)
- Number of bathrooms
- Number of balconies
- Facing direction
- Location
- Property type
- Furnishing status

The target variable is `Furnished_clean`, which categorizes properties as:

- Furnished
- Semi-Furnished
- Unfurnished

The dataset was cleaned and prepared before model training. Rows with missing target values were excluded from the classification task.

## 4. Methodology

### 4.1 Data Cleaning and Preparation

The dataset was prepared for machine learning by working with numerical and categorical features separately.

The target variable was `Furnished_clean`. Records without a target value were removed before splitting the data.

### 4.2 Feature Selection

The following seven features were selected for the classification model:

| Feature | Description |
|---|---|
| `Area_num` | Property area |
| `BHK_num` | Number of bedrooms |
| `Bathrooms_num` | Number of bathrooms |
| `Balconies_num` | Number of balconies |
| `Facing` | Property facing direction |
| `Location` | Property location |
| `Property Type` | Type of property |

### 4.3 Data Preprocessing

A preprocessing pipeline was created using Scikit-learn.

- **Numerical features:** Missing values were filled using the median, followed by standardisation.
- **Categorical features:** Missing values were filled using the most frequent value, followed by one-hot encoding.
- **Train-test split:** 80% of the data was used for training and 20% for testing.
- **Stratification:** The split preserved the approximate proportions of the furnishing categories.

### 4.4 Model Training

A Decision Tree classifier was trained using Scikit-learn with a maximum depth of 5.

Different tree depths were tested to understand the relationship between model complexity and performance.

## 5. Model Evaluation

The Decision Tree classifier was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| Baseline accuracy | 43.8% |
| Decision Tree test accuracy | 44.4% |
| Weighted ROC-AUC | Approximately 0.577 |

The baseline accuracy represents the performance of a simple model that predicts the most frequent class. The Decision Tree performed only slightly better than this baseline.

### Classification Performance

The model performed unevenly across the three furnishing categories:

- **Semi-Furnished:** The model identified most properties in this category.
- **Furnished:** The model struggled to identify this category correctly.
- **Unfurnished:** The model also showed limited ability to identify these properties.

This demonstrates why accuracy alone is not sufficient to evaluate a classification model. Precision, recall, F1-score, and ROC-AUC provide additional information about its performance.

## 6. Feature Importance

Permutation importance was used to investigate how much each feature contributed to the model's macro F1-score.

The results indicated that:

1. **Property area (`Area_num`)** was the most influential feature.
2. **Facing direction (`Facing`)** was the second most influential feature.
3. Balconies had very little measurable influence.
4. Location and BHK had approximately zero importance.
5. Property type and bathrooms had slightly negative importance scores.

Negative permutation importance does not necessarily mean that a feature is inherently harmful. It indicates that shuffling the feature did not reduce the model's measured performance and may have slightly improved it.

## 7. Key Findings

- Real-world property data requires cleaning and preprocessing before it can be used effectively for machine learning.
- Numerical and categorical variables require different preprocessing techniques.
- The Decision Tree achieved 44.4% test accuracy, only slightly above the baseline of 43.8%.
- The model performed much better on Semi-Furnished properties than on the other two categories.
- Property area and facing direction were the most influential features in the feature-importance analysis.
- The current feature set and model did not provide sufficiently strong predictions for reliable classification.

## 8. Limitations and Future Improvements

The results suggest several possible improvements:

- Explore additional features that may better explain furnishing status.
- Investigate class imbalance and compare different strategies for handling it.
- Compare the Decision Tree with other classifiers, such as Random Forest and Logistic Regression.
- Tune model hyperparameters using cross-validation.
- Examine confusion matrices and class-specific precision, recall, and F1-scores.
- Evaluate whether additional data cleaning or feature engineering improves performance.

Any future model should be evaluated against the same baseline and test set to make comparisons meaningful.

## 9. Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git and GitHub

## 10. Conclusion

This project demonstrates a supervised machine learning workflow using a real-world property rental dataset. It covers data preparation, feature selection, preprocessing, Decision Tree training, model evaluation, and feature-importance analysis.

Although the model achieved only a modest improvement over the baseline, the exercise highlights the importance of evaluating class-specific performance and understanding which features contribute to predictions.

The results also show that selecting a suitable model and evaluation metric is essential when applying machine learning to real-world datasets.