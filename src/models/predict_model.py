import pickle
import os
import pandas as pd

from src.features.build_features import set_categorical, transform_columns

def load_model(name):
    # Get the current directory of the predict_model.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the name file using relative paths
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'models'))
    model_file_path = os.path.join(data_dir, name)

    # Load a saved pickle model
    loaded_model = pickle.load(open(model_file_path, "rb"))
    return loaded_model

def make_predictions(model, X_test):
    X_test = transform_columns(X_test)
    X_test = set_categorical(X_test)
    # Make a predictions
    preds = model.predict(X_test)
    return preds

def prepare_and_save_predictions(preds, X_test_passengerId, name):
    # Prepare the data for Kaggle
    preds_df = pd.DataFrame(columns=['Transported'], data=preds, dtype=bool)
    preds_df['PassengerId'] = X_test_passengerId
    # Switch the columns
    transported_col = preds_df['Transported']  # Store the 'Transported' column
    preds_df.drop(columns=['Transported'], inplace=True)  # Drop the 'Transported' column from the DataFrame
    preds_df['Transported'] = transported_col  # Add the 'Transported' column back to the DataFrame, but in a different position
    


    # Get the current directory of the predict_model.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the name file using relative paths
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'models'))
    predictions_file_path = os.path.join(data_dir, name)

    # Save the table into csv file
    preds_df.to_csv(predictions_file_path, index=False)