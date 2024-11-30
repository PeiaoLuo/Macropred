# Macroeconomic Index Prediction Project

This project focuses on predicting macroeconomic indices. 
The approach involves forecasting the difference in the actual values of the indices and subsequently converting the results into year-over-year (YoY) format.

## Workflow

### 1. Feature Selection with `Engineer.py`
The script `Engineer.py` selects features from the feature table located in the `tables` directory for specific targets. 
It calculates static evaluation metrics such as the Maximal Information Coefficient (MIC) to assign weighted scores to features and selects a specified number of features based on these scores.

### 2. Modeling with `ols_ts.py`
The script `ols_ts.py` applies various preprocessing methods, including:
- Scaling
- Handling multicollinearity
- Managing seasonality
- Resampling  

It then uses Ordinary Least Squares (OLS) regression to predict the difference in values. 
This script provides both fitted and predicted results for the target indices.

### 3. Visualization and Evaluation with `plot_and_eval.py`
The script `plot_and_eval.py` generates visualizations of the results obtained in the previous step. Additionally, 
it processes the data to convert it into year-over-year (YoY) format and evaluates the predictions accordingly.
