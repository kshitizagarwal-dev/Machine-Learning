# Machine-Learning #Projects of  machine learning
This is the Boston House-Prediction Project.
Dataset is taken from the sklearn.datasets.
First, we convert the dataset into the pandas DataFrame.
Then, do the EDA on the dataset- correlation analysis and uses heatmap for this purpose.
 Convert the datatype of the features.
Here, we use Linear Regression, DecisionTreeRegressor, Random Forest Regressor.
We have created 20 models in total and 8 models of each algorithm mention above.
We created feature sets by doing correlation analysis, Standard Scaling, Normalization, and automatic Hyperparameter Tuning. 
For automatic Hyperparameter Tuning, we use GridSearchCv on RandomForestRegressor, DecisionTreeRegressor.
For the purpose of evaluation we use - r2 matrix and mean_ablsolute_error. 
Conclusion -----> rf_model_3 created using StandardScaling dataset with all 13 features, is best suited for this dataset.
