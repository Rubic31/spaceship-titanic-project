import pandas as pd
import numpy as np

# Preprocessing tools
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from pandas.api.types import CategoricalDtype

def split_cabin_column(data):
    """
    Splits cabin column into Deck and Side columns. Removes num from cabin column
    and drops cabin column altogether.

    Args:
    data (pandas.DataFrame): DataFrame containing Cabin column in a specific format: deck/num/side (str).

    Returns:
    pandas.DataFrame: DataFrame with Cabin column dropped and 2 new columns added (Deck and Side).
    """
    # Split the 'Cabin' column by '/' and extract the first element to create a new 'Deck' column
    data['Deck'] = data['Cabin'].str.split('/').str[0]

    # Split the 'Cabin' column by '/' and extract the third element to create a new 'Side' column
    data['Side'] = data['Cabin'].str.split('/').str[2]

    # Drop the 'Cabin' column from the dataframe
    data.drop(columns="Cabin", inplace=True)
    return data

def transform_passengerId_column(data):
    """
    Extracts groups from PassengerId column and puts in new column named GroupNumber. 
    Counts occurrences of each GroupNumber and creates new column named GroupSize based on counts.
    Drops PassengerId column.

    Args:
    data (pandas.DataFrame): DataFrame containing PassengerId column in a specific format: gggg_pp (str).

    Returns:
    pandas.DataFrame: DataFrame with PassengerId column dropped and 2 new columns added (GroupNumber, GroupSize).
    """
    # Split the 'PassengerId' column by '_' and extract the first element to create a new 'GroupNumber' column
    data['GroupNumber'] = data['PassengerId'].str.split('_').str[0]

    # Drop the 'PassengerId' column from the dataframe
    data.drop(columns="PassengerId", inplace=True)

    # Count the occurrences of each GroupNumber
    group_counts = data['GroupNumber'].value_counts()

    # Create a new column 'GroupSize' based on the counts
    data['GroupSize'] = data['GroupNumber'].map(group_counts)
    return data

def drop_name_column(data):
    """
    Drops Name column.

    Args:
    data (pandas.DataFrame): DataFrame containing Name column.

    Returns:
    pandas.DataFrame: DataFrame with Name column dropped.
    """
    # Drop the 'Name' column from the dataframe
    data.drop(columns="Name", inplace=True)

    return data

def sum_spending_columns(data):
    """
    Calculate the total spending by summing up individual spending columns. Will temporarly fill NaN in spending columns
    to avoid NaN in SpendingTotal column. Fills median of corresponding column.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing RoomService, FoodCourt, ShoppingMall, Spa and VRDeck columns.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'SpendingTotal' column representing the sum of individual spending columns.
    """
    # Summing up individual spending columns to get the total spending
    # Using fillna(median) so that if one argument is NaN it will be replaced by median (equal to 0)
    # Therefore SpendingTotal will not be NaN
    data['SpendingTotal'] = (
        data['RoomService'].fillna(data['RoomService'].median()) + data['FoodCourt'].fillna(data['FoodCourt'].median()) +
        data['ShoppingMall'].fillna(data['ShoppingMall'].median()) + data['Spa'].fillna(data['Spa'].median()) + 
        data['VRDeck'].fillna(data['VRDeck'].median())
    )
    
    return data

def calculate_group_spending(data):
    """
    Calculate total spending and average spending per group member.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing spending information.

    Returns:
    - pd.DataFrame: DataFrame with additional columns:
        - 'GroupTotalSpending': Total spending of the group (sum of SpendingTotal for all group members).
        - 'AvgSpendingPerMember': Average spending per group member.
    """
    # Calculate total spending of the group
    data['GroupTotalSpending'] = data.groupby('GroupNumber')['SpendingTotal'].transform('sum')

    # Calculate average spending per group member
    data['AvgSpendingPerMember'] = data['GroupTotalSpending'] / data['GroupSize']

    return data

def transform_columns(data):
    """
    Transforms DataFrame by adding, splitting or removing some columns.

    Args:
    data (pandas.DataFrame): DataFrame containing Name, Cabin and PassengerId column in a specific formats: 
    Name: First Last (str),
    Cabin: deck/num/side (str),
    PassengerId: gggg_pp (str),
    RoomService: (int/float - numeric),
    FoodCourt: (int/float - numeric),
    ShoppingMall: (int/float - numeric),
    Spa: (int/float - numeric),
    VRDeck: (int/float - numeric).

    Returns:
    pandas.DataFrame: DataFrame with Name, Cabin and PassengerId column dropped and 7 new columns added: 
    GroupNumber, GroupSize, Deck, Side, SpendingTotal, GroupTotalSpending, AvgSpendingPerMember.
    """
    drop_name_column(data)
    transform_passengerId_column(data)
    split_cabin_column(data)
    sum_spending_columns(data)
    calculate_group_spending(data)

    return data

def create_preprocessor():
    """
    Create a preprocessor for handling different types of features in a machine learning pipeline.

    The preprocessor includes separate transformations for categorical, ordinal, numeric, and boolean features.
    Categorical features are imputed with the most frequent value and one-hot encoded.
    Ordinal features are imputed with the most frequent value.
    Numeric features are imputed with the median and scaled using RobustScaler.
    Boolean features are imputed with the most frequent value.

    Returns:
    - ColumnTransformer: Preprocessor containing separate transformers for different feature types.
    """
    # Define different features and transformer pipelines
    categorical_features = ["HomePlanet", 'Destination', 'GroupNumber', 'Deck', 'Side']
    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    ordinal_features = ['GroupSize']
    ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))])

    numeric_features = ["Age", 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'SpendingTotal', 'GroupTotalSpending', 'AvgSpendingPerMember']
    numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", RobustScaler())]) # Good for handling outliers

    boolean_features = ['CryoSleep', 'VIP']  # Define boolean features

    # Handle missing values for boolean features
    boolean_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))])


    # Setup preprocessing steps (fill missing values, then convert to numbers)
    preprocessor = ColumnTransformer(
                transformers=[
                            ("cat", categorical_transformer, categorical_features),
                            ("ord", ordinal_transformer, ordinal_features),
                            ("num", numeric_transformer, numeric_features), 
                            ("bool", boolean_transformer, boolean_features)])
    return preprocessor

def set_categorical(data):
    """
    Convert specified columns in a DataFrame to categorical data types.

    Parameters:
    - data (pd.DataFrame): Input DataFrame to be modified.

    Returns:
    - pd.DataFrame: DataFrame with specified columns converted to categorical data types.
    """
    groupsize_cat_dtype = CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8], ordered=True)
    data['GroupSize'] = data['GroupSize'].astype(groupsize_cat_dtype)
    data['GroupNumber'] = data['GroupNumber'].astype('category')
    data["HomePlanet"] = data["HomePlanet"].astype("category")
    data["CryoSleep"]= data["CryoSleep"].astype("category")
    data["Destination"]= data["Destination"].astype("category")
    data["VIP"]= data["VIP"].astype("category")
    data["Deck"]= data["Deck"].astype("category")
    data["Side"]= data["Side"].astype("category")
    return data