# Insurance Charges Prediction

This project analyzes the `insurance.csv` dataset and builds a multiple linear regression model to predict individual medical insurance charges using demographic and health-related features. This dataset is the Medical Insurance Cost Dataset found on Kaggle.

The notebook explores the dataset, cleans the data, performs exploratory analysis, engineers a BMI-smoking interaction term, trains a regression model, evaluates its performance, and interprets the main drivers of predicted charges.

## Project objective

The goal of this project is to answer two questions:

- Which factors are most strongly associated with higher medical insurance charges?
- How well can a linear regression model predict insurance charges from the available variables?

The target variable is:

- `charges`: individual medical insurance cost

The input features are:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

## Dataset

The project uses the `insurance.csv` dataset, which contains personal and regional attributes together with medical insurance charges.

After inspection, the data was cleaned by removing one duplicate row and confirming that there were no missing values.

## Workflow

The notebook follows this structure:

1. Data loading and inspection.
2. Data cleaning and validation.
3. Descriptive statistics.
4. Exploratory data analysis.
5. Feature engineering and preprocessing.
6. Linear regression model training.
7. Model evaluation.
8. Coefficient interpretation.
9. Final conclusion.

## Key analysis insight

One of the most important findings is that smokers and non-smokers do not follow the same BMI-charge relationship.

When all observations are plotted together, the overall trend can hide an important pattern: smokers tend to have much higher insurance charges, and the relationship between BMI and charges is much steeper for smokers than for non-smokers.

Because of this, the model includes an interaction term:

- `bmi_smoker_inter = bmi * smoker_yes`

This allows the model to represent different BMI effects for smokers and non-smokers instead of forcing a single shared slope.

## Model

The project uses **multiple linear regression** with one-hot encoded categorical variables.

Categorical preprocessing includes:

- `sex` encoded with `drop_first=True`
- `smoker` encoded with `drop_first=True`
- `region` encoded with `drop_first=True`

An additional interaction feature is created:

- `bmi_smoker_inter`

The train-test split uses:

- `test_size = 0.30`
- `random_state = 42`

## Evaluation metrics

The notebook reports the following regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (`R^2`)

It also includes an **actual vs predicted** scatter plot to visually assess model fit.

## Main findings

The analysis shows that:

- Smoking status is the strongest predictor of insurance charges.
- BMI becomes much more important when the person is a smoker.
- Age has a positive relationship with charges.
- Region has a comparatively smaller effect than smoking-related variables.

This means the model is more informative when it captures the interaction between health risk factors rather than treating each variable as fully independent.

## Files

- `insurance.ipynb` - full cleaned notebook
- `insurance.csv` - dataset file required to run the notebook

## How to run

1. Place `insurance.csv` in the same folder as the notebook.
2. Open `insurance.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Run all cells from top to bottom.

## Tools and libraries

The notebook uses:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Possible next steps

There are several natural extensions to this project:

- Compare linear regression with tree-based models such as Random Forest or XGBoost.
- Test whether a log transformation of `charges` improves performance.
- Add residual analysis for stronger model diagnostics.
