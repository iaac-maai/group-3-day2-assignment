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