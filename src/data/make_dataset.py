import pandas as pd
import os

def load_raw_train():
    """
    Function to load the train.csv file from a data/raw folder into a pandas DataFrame.

    Returns:
    pandas.DataFrame: DataFrame containing the train data.
    """
    # Get the current directory of the make_dataset.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the train.csv file using relative paths
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'raw'))
    train_file_path = os.path.join(data_dir, 'train.csv')
    
    # Load the train.csv file into a pandas DataFrame
    train_data = pd.read_csv(train_file_path)
    
    return train_data