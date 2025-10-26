1. Project Name 

    Title: California Housing Price Prediction Using AI Methods

2. Project Overview

    This project focuses on applying and comparing various machine learning techniques, including individual regression models (Decision Tree, k-Nearest Neighbors, Support Vector Machine) and ensemble methods (Voting, Stacking, Bagging), to predict house prices in the California Housing dataset. 
    The primary goal is to maximize the prediction accuracy of the target variable MedHouseVal by optimizing models, evaluating their performance, and leveraging ensemble techniques.

3. Data Source or Dataset

    Dataset: California Housing Dataset
    Source: California Housing dataset, accessed via sklearn.datasets.fetch_california_housing().
    
    <a>https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset </a>
    
    Description: The dataset contains 20,640 entries with 9 features (e.g., MedInc, HouseAge, AveRooms) and the target variable MedHouseVal (median house value in $100,000s). No missing values were present initially.

4. Tools

    jupyter notebook

    Programming Language: Python
    
    Libraries:

    numpy, pandas (data manipulation)
    
    matplotlib, seaborn (visualization)
    
    sklearn (model training, evaluation, and data preprocessing)
    
    xgboost (gradient boosting) . 
    
    Techniques: GridSearchCV (hyperparameter tuning), 5-fold cross-validation

5. KPI (Key Performance Indicators / Questions)

   Primary KPIs:
   
   1. Test R² Score (generalization ability)
   
   2. Test Mean Squared Error (MSE) (prediction accuracy)   
   
   3. Cross-Validation MSE (model stability)


    Questions:

    . Which model achieves the highest prediction accuracy for MedHouseVal?
    
    . How do ensemble models compare to individual regression models?
    
    . What impact does data cleaning have on model performance?



6. Data Cleaning / Preparation

   Steps:

   1- Removed entries with MedHouseVal ≥ 5.0 (reduced to 19,648 entries).
   
   2- Removed upper 2% outliers (reduced to 17,145 entries).
   
   3- Applied IQR method (reduced to 15,149 entries).
   
   4- Used Isolation Forest for final outlier removal (reduced to 14,391 entries).


   Results: No duplicates found; all features normalized; 30.3% data reduction improved quality.

7. Exploratory Data Analysis

     Observations:
   
     1- MedHouseVal showed a cut-off at 5.0, indicating potential outliers.
   
     2- Features like MedInc (median income) were strong predictors.
   
     3- Initial distribution skewed, requiring outlier removal and normalization.


     Visualizations: <a>[Visualizations.png](https://github.com/amer-deiri/Optimizing-the-house-price-forecast-in-California/blob/main/Visualizations.png) </a>

8. Data Analysis

   Models Evaluated:

   1. Individual: Linear Regression (LASSO), Decision Tree, KNN, SVR, Random Forest, XGBoost, Neural Network.
   
   2. Ensemble: Voting (Weighted), Stacking (RF/LR/Ridge Meta), Bagging (RF Base).


   Optimization: Hyperparameters tuned using GridSearchCV with 5-fold cross-validation.

   Example: XGBoost (learning_rate=0.05, max_depth=8, n_estimators=300).


   Metrics: Training R², Test R², Test MSE, Cross-Validation MSE.

   Results:

   1. Best Individual Model: XGBoost (Test R²: 0.8207, Test MSE: 0.1446).
   
   2. Best Ensemble Model: Stacking (LR Meta) and (Ridge Meta) (Test R²: 0.8237, Test MSE: 0.1421).



9. Results / Findings

   Model Performance:

   1- Best Individual Model: XGBoost (Test R²: 0.8207, Test MSE: 0.1446).
   
   2- Best Ensemble Model: Stacking (LR/Ridge Meta) (Test R²: 0.8237, Test MSE: 0.1421, CV MSE: 0.2609).
   
   3- Weakest Models: Decision Tree (Test R²: 0.6123), Bagging (Test R²: 0.7464).
   
   Conclusion: Ensemble models, especially Stacking, significantly outperform individual models .


   Visualizations:

   Barplots: Stacking (LR Meta) and (Ridge Meta) lead in Test R² and lowest MSE.
   KDE Plot: Stacking (LR Meta) shows narrowest error distribution.
   Scatter Plots: Stacking (LR Meta) aligns closest to the ideal line.



10. Recommendations

    Improvements:

    1- Explore advanced dimensionality reduction (e.g., PCA) or further outlier removal.
    
    2- Implement additional boosting models (e.g., LightGBM).
    
    3- Construct new features (e.g., area in square kilometers or population density).
    
    4- Refine hyperparameters with Bayesian optimization.
    
    5- Investigate log transformation effects on specific models.


    Future Research: Focus on advanced ensemble techniques and enhanced feature engineering.

11. Limitations:

    1- Computational Resources: GridSearchCV was time-intensive, limiting hyperparameter exploration.

    2- Dataset Size: Reduction to 14,391 entries may lose some data variability.

    3- Model Complexity: Overfitting observed in KNN and Decision Tree, requiring further regularization (Training R² near 1.0).

    4- Data Reduction: Significant data loss (30.3%) may affect representativeness.

    5- Feature Scope: Limited to existing features; additional external data could enhance predictions.

    6- Unexplored Techniques: Log transformation and additional models not fully tested.

12. References

    1. California Housing Dataset: scikit-learn Documentation
    https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
    
    2. scikit-learn Library: https://scikit-learn.org/stable/
    
    3. XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/

    
