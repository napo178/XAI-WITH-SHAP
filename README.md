# XAI-WITH-SHAP

# DATA:
https://drive.google.com/file/d/1F_NVdj4YtbscMHYyUQF81u7DTSwMul9W/view?usp=sharing

# Problem:
Hepamine - A Liver Disease Microarray Database, Visualization Platform and Data-Mining Resource


# shap:

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see papers for details and citations).
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way






Tree ensemble example (XGBoost/LightGBM/CatBoost/scikit-learn/pyspark models)
While SHAP can explain the output of any machine learning model, we have developed a high-speed exact algorithm for tree ensemble methods (see our Nature MI paper). Fast C++ implementations are supported for XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models:

import xgboost
import shap

# Train XGBoost model
X, y = shap.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# Explain the model's predictions using SHAP

(same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
