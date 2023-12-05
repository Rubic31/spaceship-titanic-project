import matplotlib.pyplot as plt
import pandas as pd

def plot_bar_for_categorical(data, categorical_features):

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

    num_features = len(numerical_features)
    
    # Calculate the number of rows and columns for subplots
    num_rows = num_features // 3 + (num_features % 3 > 0)
    num_cols = min(num_features, 3)
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Create a grid of subplots

    # Flatten the axes array if there is only one row
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