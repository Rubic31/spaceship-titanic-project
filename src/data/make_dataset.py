import pandas as pd
import os

def load_data(path):
    """
    Load a dataset from a specified file path into a pandas DataFrame.
    Parameters:
    - path (str): Relative path to the dataset file in a format: type_data/name.csv
    where type_data is either raw, interim or processed.

    Returns:
    - pd.DataFrame: Loaded dataset in a pandas DataFrame.
    """
    # Get the current directory of the make_dataset.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the train_transformed.csv file using relative paths
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    file_path = os.path.join(data_dir, path)
    
    # Load the train.csv file into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    return data