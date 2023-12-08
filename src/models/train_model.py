import numpy as np
import pandas as pd

# For saving and loading models
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pandas.api.types import CategoricalDtype

from src.features.build_features import set_categorical

def split_X_y(data, target):
    """
    Split a DataFrame into features (X) and a target feature (y).

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing features and the target feature.
    - target (str): The name of the target feature to be separated from the features.

    Returns:
    - pd.DataFrame: DataFrame containing only the features (X).
    - pd.Series: Series containing the target feature (y).
    """
    # Extract features (X) by dropping the target column
    x = data.drop(target, axis=1)
    # Extract the target feature (y)
    y = data[target]
    return x, y

def models_cross_val(models, X, y, preprocessor): 
    """
    Perform cross-validation for multiple machine learning models.

    Parameters:
    - models (dict): A dictionary containing model names as keys and corresponding model instances as values.
    - X (pd.DataFrame): Input features for the machine learning models.
    - y (pd.Series): Target variable for the machine learning models.
    - preprocessor (ColumnTransformer): Preprocessor for handling feature transformations.

    Returns:
    - dict: A dictionary containing model names as keys and their cross-validated mean scores as values.
    """
    # Random seed for reproducible results
    np.random.seed(42)

    # Set dtype of GroupSize to ordinal, and GroupNumber to categorical
    groupsize_cat_dtype = CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8], ordered=True)
    X['GroupSize'] = X['GroupSize'].astype(groupsize_cat_dtype)
    X['GroupNumber'] = X['GroupNumber'].astype('category')

    # Make a list to keep model scores
    model_scores = {}
        
    # Loop through models
    for name, model in models.items():
        
        # Create a preprocessing and modelling pipeline
        clf_pipe = Pipeline(steps=[("preprocessor", preprocessor),
                        (name, model)])
        
        # Evaluate the model and append its score to model_scores
        model_scores[name] = cross_val_score(clf_pipe, X, y, cv=5, n_jobs=-1).mean()
        
    return model_scores

def model_find_hyperparameters(X, y, param_distributions, n_iter):
    """
    Perform hyperparameter tuning for an XGBoost classifier using RandomizedSearchCV.

    Parameters:
    - X (pd.DataFrame): Input features for the machine learning model.
    - y (pd.Series): Target variable for the machine learning model.
    - param_distributions (dict): Dictionary specifying the hyperparameter distributions for RandomizedSearchCV.
    - n_iter (int): Number of parameter settings that are sampled.

    Returns:
    - RandomizedSearchCV: Fitted RandomizedSearchCV instance with the best hyperparameters.
    """
    # Set the random seed for reproducibility
    np.random.seed(42)
    X = set_categorical(X)
    # Setup RandomizedSearchCV since GridSearchCV would take too much time
    gs_clf = RandomizedSearchCV(estimator=XGBClassifier(tree_method="approx", enable_categorical=True), 
                                n_iter=n_iter,
                                param_distributions=param_distributions,
                                cv=5, # 5-fold cross-validation
                                verbose=0, n_jobs=-1)

    # Fit the training data to grid search
    gs_clf.fit(X, y)
    return gs_clf

def train_and_save_xgboost(X, y, best_params):
    """
    Train an XGBoost classifier with specified hyperparameters and save the trained model to a file.

    Parameters:
    - X (pd.DataFrame): Input features for training the XGBoost model.
    - y (pd.Series): Target variable for training the XGBoost model.
    - best_params (dict): Dictionary containing the best hyperparameters for the XGBoost model.

    Returns:
    - XGBClassifier: Trained XGBoost classifier.
    """
    np.random.seed(42)
    X = set_categorical(X)
    # Set up a XGBoost model with best parameters
    model = XGBClassifier(subsample=best_params['subsample'], gamma=best_params['gamma'], reg_lambda=best_params['reg_lambda'], 
                          reg_alpha=best_params['reg_alpha'], n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], 
                          learning_rate=best_params['learning_rate'], colsample_bytree=best_params['colsample_bytree'], 
                          min_child_weight=best_params['min_child_weight'], tree_method="approx", enable_categorical=True)

    # Fit all the data into XGBoost model
    model.fit(X, y)

    # Get the current directory of the train_model.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the xgboost_model_1.pkl file using relative paths
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'models'))
    model_file_path = os.path.join(data_dir, 'xgboost_model_1.pkl')
    
    # Save an existing model to file
    pickle.dump(model, open(model_file_path, "wb"))

    return model