import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# Function importing Dataset
def importdata(file_path):
    """
    Function to import data from a CSV file.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The dataset as a DataFrame.
    """
    logging.info("Importing data from %s", file_path)
    balance_data = pd.read_csv(file_path)
    
    # Printing the dataset shape and observations
    logging.info("Dataset Length: %d", len(balance_data))
    logging.info("Dataset Shape: %s", str(balance_data.shape))
    logging.info("Dataset: \n%s", balance_data.to_string(index=False))
    return balance_data

# Function to calculate entropy
def calculate_entropy(p_values):
    """
    Function to calculate entropy based on given probabilities.

    Args:
    p_values (array-like): Probabilities.

    Returns:
    float: Calculated entropy.
    """
    return -np.sum(p_values * np.log2(p_values + np.finfo(float).eps))

# Function to calculate entropy for an attribute considering 'yes' and 'no' values
def calculate_entropy_attribute(df, attribute):
    """
    Function to calculate entropy for an attribute considering 'yes' and 'no' values.

    Args:
    df (DataFrame): The dataset as a DataFrame.
    attribute (str): The attribute for which entropy is to be calculated.

    Returns:
    float: Calculated entropy for the attribute.
    """
    total_instances, values, class_attribute_name = calculate_common_variables(df, attribute)

    entropy = 0

    for value in values:
        df_filtered = df[df[attribute] == value]
        instances_with_value = len(df_filtered)
        
        class_distribution = df_filtered[class_attribute_name].value_counts() / instances_with_value

        entropy_value = calculate_entropy(class_distribution)

        entropy -= (instances_with_value / total_instances) * entropy_value

    return abs(entropy)

# Function to calculate common variables
def calculate_common_variables(df, attribute):
    total_instances = len(df)
    values = df[attribute].unique()
    class_attribute_name = df.columns[-1]
    
    return total_instances, values, class_attribute_name

# Function to calculate gain information for an attribute
def calculate_gain_info(entropy_D, df, attribute):
    """
    Function to calculate gain information for an attribute.

    Args:
    entropy_D (float): Entropy of the entire dataset.
    df (DataFrame): The dataset as a DataFrame.
    attribute (str): The attribute for which gain information is to be calculated.

    Returns:
    float: Calculated gain information for the attribute.
    """

    entropy_A = calculate_entropy_attribute(df, attribute)
    gain_info = entropy_D - entropy_A

    return gain_info

# Function to calculate Gini impurity for an attribute considering 'yes' and 'no' values
def calculate_gini_impurity_attribute(df, attribute):
    """
    Function to calculate Gini impurity for an attribute considering 'yes' and 'no' values.

    Args:
    df (DataFrame): The dataset as a DataFrame.
    attribute (str): The attribute for which Gini impurity is to be calculated.

    Returns:
    float: Calculated Gini impurity for the attribute.
    """
    total_instances, values, class_attribute_name = calculate_common_variables(df, attribute)

    gini_impurity = 0

    for value in values:
        instances_with_value = len(df[df[attribute] == value])
        
        class_distribution = df[df[attribute] == value][class_attribute_name].value_counts() / instances_with_value

        gini_value = 1.0 - sum(np.square(class_distribution))

        gini_impurity -= (instances_with_value / total_instances) * gini_value

    return abs(gini_impurity)

# Driver code
def main():
    # Building Phase
    data = importdata('YOUR-FILE-PATH')
    attribute_names = data.columns[1:-1] 
    class_attribute_name = data.columns[-1] 

    # Calculate entropy for the entire dataset (Entropy D)
    class_distribution = data[class_attribute_name].value_counts() / len(data)
    entropy_D = calculate_entropy(class_distribution)

    # Calculate and print metrics for each attribute
    logging.info("\nMetrics for each attribute:")
    for attribute in attribute_names:
        entropy_val = calculate_entropy_attribute(data, attribute) 
        gain_info = calculate_gain_info(entropy_D, data, attribute ) 
        gini_impurity_val = calculate_gini_impurity_attribute(data, attribute) 

        logging.info(f"Attribute: {attribute}")
        logging.info(f"Calculated Entropy: {entropy_val:.3f}")
        logging.info(f"Calculated Gain Information: {gain_info:.3f}")
        logging.info(f"Calculated Gini Impurity: {gini_impurity_val:.3f}")
        logging.info("\n")

# Calling main function
if __name__ == "__main__":
    main()
