import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_bar_for_categorical(data, categorical_features):
    """
    Plot bar plots for categorical features in the given DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - categorical_features (list): List of column names representing the categorical features.

    Returns:
    None

    The function creates a grid of subplots with bar plots for each categorical feature.
    The number of rows and columns in the grid is determined based on the number of features.
    """

    num_features = len(categorical_features)
    
    # Calculate the number of rows and columns for subplots
    num_rows = num_features // 3 + (num_features % 3 > 0)
    num_cols = min(num_features, 3)
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Create a grid of subplots

    # Flatten the axes array if there is only one row
    if num_rows > 1:
        ax = ax.flatten()

    for i, feature in enumerate(categorical_features):

        # If there's only one row, ax is not a list, so use a different indexing approach
        if num_rows == 1:
            current_ax = ax[i % 3] if num_cols > 1 else ax
        else:
            current_ax = ax[i]

        # Plot bar plots for each categorical feature
        value_counts = data[feature].value_counts()
        current_ax.bar(
            data[feature].value_counts().index,
            data[feature].value_counts().values,
            color='C{}'.format(i), alpha=0.5
        )
        current_ax.set_title(feature)

        # Replace x-axis labels with "True" and "False" for binary variables
        if len(value_counts) == 2 and all(value in [True, False] for value in value_counts.index):
            current_ax.set_xticks([False, True])
            current_ax.set_xticklabels(['False', 'True'])

    # Set y label for all plots
    fig.text(0.001, 0.5, 'Count', va='center', rotation='vertical')
    
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()


def plot_hist_for_numerical(data, numerical_features):
    """
    Plot histograms for numerical features in the given DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - numerical_features (list): List of column names representing the numerical features.

    Returns:
    None

    The function creates a grid of subplots with histograms for each numerical feature.
    The number of rows and columns in the grid is determined based on the number of features.
    """
    num_features = len(numerical_features)
    
    # Calculate the number of rows and columns for subplots
    num_rows = num_features // 3 + (num_features % 3 > 0)
    num_cols = min(num_features, 3)
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Create a grid of subplots

    # Flatten the axes array if there is more than one row
    if num_rows > 1:
        ax = ax.flatten()

    for i, feature in enumerate(numerical_features):
        
        # If there's only one row, ax is not a list, so use a different indexing approach
        if num_rows == 1:
            current_ax = ax[i % 3] if num_cols > 1 else ax
        else:
            current_ax = ax[i]

        # Plot hist plots for each categorical feature
        current_ax.hist(
            data[feature],bins=10,
            color='C{}'.format(i), alpha=0.5
        )
        current_ax.set_title(feature)

    # Set x and y label for all plots
    fig.text(0.001, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.text(0.5, 0.001, 'Values', ha='center')
    
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()

def plot_boxplots(df, selected_data, target_feature):
    """
    Plot boxplots for selected features based on the binary target feature in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - selected_data (list): List of column names representing the features to plot.
    - target_feature (str): The binary target column based on which the data will be categorized for boxplots.

    Returns:
    None

    The function creates subplots with boxplots for each selected feature, categorized by the binary target feature.
    """

# Calculate the number of rows and columns for subplots
    num_rows = len(selected_data)
    num_cols = 2

    # Create a DataFrame with selected columns
    df_selected = df[[target_feature] + selected_data]

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 5 * num_rows))

    for i, feature in enumerate(selected_data):
        # Filter the DataFrame for each category of target feature
        target_false = df_selected[df_selected[target_feature] == False]
        target_true = df_selected[df_selected[target_feature] == True]

        # Determine subplot position
        row_position = i


        if num_rows > 1:
            # Create boxplot for "False" in target column
            axes[row_position, 0].boxplot(
                target_false[target_false[feature].notnull()][feature]
            )
            axes[row_position, 0].set_title(f'Boxplot of {feature} for False {target_feature}')
            axes[row_position, 0].set_xticklabels([''])
            axes[row_position, 0].set_ylim([0, df_selected[feature].max() * 1.1])
            axes[row_position, 0].set_ylabel('Values')

            # Create boxplot for "True" in target column
            axes[row_position, 1].boxplot(
                target_true[target_true[feature].notnull()][feature]
            )
            axes[row_position, 1].set_title(f'Boxplot of {feature} for True {target_feature}')
            axes[row_position, 1].set_xticklabels([''])
            axes[row_position, 1].set_ylim([0, df_selected[feature].max() * 1.1])
            axes[row_position, 1].set_ylabel('Values')
        else:
            # Create boxplot for "False" in target column
            axes[0].boxplot(
                target_false[target_false[feature].notnull()][feature]
            )
            axes[0].set_title(f'Boxplot of {feature} for False {target_feature}')
            axes[0].set_xticklabels([''])
            axes[0].set_ylim([0, df_selected[feature].max() * 1.1])
            axes[0].set_ylabel('Values')

            # Create boxplot for "True" in target column
            axes[1].boxplot(
                target_true[target_true[feature].notnull()][feature]
            )
            axes[1].set_title(f'Boxplot of {feature} for True {target_feature}')
            axes[1].set_xticklabels([''])
            axes[1].set_ylim([0, df_selected[feature].max() * 1.1])
            axes[1].set_ylabel('Values')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()

def plot_correlation_matrix(dataframe, numerical_features, target_feature):
    """
    Plots the correlation matrix for a given DataFrame and list of numerical features and target feature.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing the numerical features.
    - numerical_features_and_target (list): List of column names representing the numerical features and target.
    - target_feature (str): Target feature name.

    Returns:
    None
    """

    # Extract numerical features and target feature from the dataframe
    df_numeric = dataframe[numerical_features + [target_feature]]
    
    # Create and plot the correlation matrix
    correlation_matrix = df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    
    # Set plot title
    plt.title('Correlation Matrix')
    
    # Show the plot
    plt.show()

def plot_crosstab(data, feature_name, target_feature):
    """
    Plot crosstab plot for categorical feature vs target feature in the given DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - feature_name (str): Name of feature column.
    - target_feature (str): Name of target feature column.

    Returns:
    None
    """
    # Create crosstab of target feature and HomePlanet
    pd.crosstab(data[feature_name], data[target_feature]).plot(kind="bar", 
                                                    figsize=(10,6), 
                                                    color=["salmon", "lightblue"])

    # Set title and y label
    plt.title(f"{target_feature} Frequency per {feature_name}")
    plt.ylabel("Amount")
    plt.xticks(rotation=0); # keep the labels on the x-axis vertical