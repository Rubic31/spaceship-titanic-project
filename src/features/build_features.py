import pandas as pd
import numpy as np

# Deducing gender from name
# import gender_guesser as gender
import nltk
from nltk.corpus import names

# Preprocessing tools
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

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

def transform_name_column(data):
    """
    Predicts sex from Name column and puts in new column named Sex. 
    Uses gender_guesser for predictions. The result will be one of: 
    unknown (name not found), andy (androgynous), male, female, 
    mostly_male, or mostly_female. The difference between andy and unknown 
    is that the former is found to have the same probability to be male than to be female, 
    while the later means that the name wasn't found in the database. Citation from https://pypi.org/project/gender-guesser/.
    Drops Name column.

    Args:
    data (pandas.DataFrame): DataFrame containing Name column in a specific format: First Last (str).

    Returns:
    pandas.DataFrame: DataFrame with Name column dropped and 1 new column added (Sex).
    """
    
    # # Download the names corpus from NLTK
    # nltk.download('names')

    # # Create a DataFrame from the NLTK names corpus
    # male_names = [(name, 'Male') for name in names.words('male.txt')]
    # female_names = [(name, 'Female') for name in names.words('female.txt')]
    # nltk_df = pd.DataFrame(male_names + female_names, columns=['Name', 'Gender'])

    # # Convert the name to uppercase for case-insensitive matching
    # data['First Name'] = data['Name'].str.split(" ").str[0]
    # # data['First Name'] = data['First Name'].str.upper()

    # # Merge with the NLTK names dataset
    # merged_df = pd.merge(data, nltk_df, how='left', left_on='First Name', right_on='Name', suffixes=('_input', '_nltk'))

    # # Create a new column "Sex" based on predictions from the "Name" column
    # merged_df['Sex'] = merged_df['Gender'].fillna('Unknown')

    # # Drop unnecessary columns
    # merged_df = merged_df.drop(['Name_nltk', 'Gender'], axis=1)

    # return merged_df

    # for nameLastName in data['Name']:

    #     name = nameLastName.str.split(" ").str[0]
    #     # Convert the name to uppercase for case-insensitive matching
    #     name = name.upper()

    #     # Check if the name is in the dataset
    #     if name in df['Name'].str.upper().values:
    #         predicted_gender = df.loc[df['Name'].str.upper() == name, 'Gender'].values[0]
    #         data['Sex'] = predicted_gender
    #     else:
    #         data['Sex'] = 'Unknown'


    # # Initialize the gender detector
    # detector = gender.Detector()

    # # Name column contains the full name in the format "First Last"
    # data['Sex'] = data['Name'].apply(lambda x: detector.get_gender(x.split()[0]))
    



    # Drop the 'Name' column from the dataframe
    data.drop(columns="Name", inplace=True)

    return data

def transform_columns(data):
    """
    Transforms DataFrame by adding, splitting or removing some columns.

    Args:
    data (pandas.DataFrame): DataFrame containing Name, Cabin and PassengerId column in a specific formats: 
    Name: First Last (str),
    Cabin: deck/num/side (str),
    PassengerId: gggg_pp (str).

    Returns:
    pandas.DataFrame: DataFrame with Name, Cabin and PassengerId column dropped and 4 new columns added: 
    GroupNumber, GroupSize, Deck, Side.
    """
    transform_name_column(data)
    transform_passengerId_column(data)
    split_cabin_column(data)
    return data