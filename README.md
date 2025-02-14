# Road Traffic Accidents

*Yarin Yerushalmi Levi, Ken Yaggel, Tal Amar, Idan Hershkovitz*  
*June 2023*

## Description

This project uses machine learning to analyze a detailed dataset of road traffic accidents from 2017–2020. The goal is to develop predictive models for accident severity, offering actionable insights for insurance companies and road safety authorities. By leveraging various classification models, the study identifies key factors influencing accident outcomes and suggests targeted interventions to enhance road safety.

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents).

## Techniques

- **Exploratory Data Analysis (EDA):**  
  In-depth analysis of accident distributions, correlations, and trends through visualizations and statistical analysis.

- **Data Preprocessing:**  
  Techniques such as missing value imputation, one-hot encoding, and splitting data into training and testing sets to ensure unbiased model evaluation.

- **Feature Selection with Chi-square Test:**  
  Evaluates the dependency between categorical features and accident severity to select the most informative features.

- **Handling Imbalanced Data:**  
  Uses undersampling, class weighting, and ensemble methods to mitigate the impact of imbalanced classes.

- **Model Selection & Hyperparameter Tuning:**  
  Applies `RandomizedSearchCV` for optimizing hyperparameters across various models like Decision Trees, XGBoost, LightGBM, Gradient Boosting, and Random Forest.  
  - [XGBoost Documentation](https://xgboost.readthedocs.io/)  
  - [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
  - [Scikit-learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

- **Model Evaluation:**  
  Uses metrics such as balanced accuracy, precision, recall, F1-score, and AUC-ROC to assess performance, with special attention to the minority class (Fatal injury).

- **Explainable AI (XAI):**  
  Implements SHAP (Shapley Additive Explanations) to interpret model decisions and reveal key feature impacts.  
  - [SHAP GitHub Repository](https://github.com/slundberg/shap)

## Libraries and Technologies

- **[PyCaret](https://pycaret.org/):**  
  Simplifies the model selection and evaluation process.

- **[Scikit-learn](https://scikit-learn.org/stable/):**  
  Provides tools for preprocessing, model training, and evaluation.

- **[XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), and [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html):**  
  Used for building and tuning classification models.

- **[SHAP](https://github.com/slundberg/shap):**  
  Enhances model interpretability by quantifying feature importance.

- **[Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/):**  
  Essential for data manipulation and numerical computation.

## Project Structure

```plaintext
.
├── notebooks/           # A single notebook containing all the code
├── reports/             # Final report, figures, and supplementary materials
└── README.md            # Project documentation (this file)
