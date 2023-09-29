# Author: Mederbek Dzhumabaev
# This script performs Apriori algorithm for frequent itemset generation and association rule mining.

from itertools import combinations
from collections import Counter
from typing import List, Dict, Any, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_SUP = 0.6
MIN_CONF = 0.8

def generate_subsets(itemset: frozenset) -> List[frozenset]:
    """Generates all subsets of an itemset.
    Args:
        itemset: A frozenset representing an itemset.
    Returns:
        A list of frozensets representing all subsets of the itemset.
    """
    subsets = []

    try:
        for i in range(len(itemset) + 1):
            for subset in combinations(itemset, i):
                subsets.append(frozenset(subset))
    except Exception as e:
        logging.error(f"An error occurred while generating subsets: {str(e)}")

    return subsets

def count_itemset_occurrences(D: List[frozenset], subsets_generator) -> Counter:
    """
    Count occurrences of itemsets in the dataset.
    Args:
        D: A list of frozensets, where each frozenset represents a transaction.
        subsets_generator: A function to generate subsets.
    Returns:
        A Counter object containing itemset occurrences.
    """
    itemset_counts = Counter()

    try:
        for transaction in D:
            subsets = subsets_generator(transaction)

            for itemset in subsets:
                itemset_counts[itemset] += 1
    except Exception as e:
        logging.error(f"An error occurred while counting itemset occurrences: {str(e)}")

    return itemset_counts

def calculate_support(count: int, total_transactions: int) -> float:
    """
    Calculate the support for an itemset.
    Args:
        count: The number of transactions in the dataset that contain the itemset.
        total_transactions: The total number of transactions.
    Returns:
        The support of the itemset.
    """
    return count / total_transactions

def set_parent_for_itemset(itemset: frozenset, itemset_table: Dict, subsets_generator):
    """
    Set the parent for the itemset in the itemset table.
    Args:
        itemset: The itemset for which to set the parent.
        itemset_table: A dictionary representing the itemset table.
        subsets_generator: A function to generate subsets.
    """
    try:
        parent = None

        for subset in subsets_generator(itemset):
            if subset in itemset_table:
                parent = subset
                break

        # Set the parent of the itemset.
        itemset_table[itemset]["parent"] = parent
    except Exception as e:
        logging.error(f"An error occurred while setting the parent for itemset {itemset}: {str(e)}")

def create_itemset_table(D: List[frozenset], MIN_SUP: float) -> Dict[frozenset, Dict[str, Any]]:
    """
    Creates a table for itemsets.
    Args:
        D: A list of frozensets, where each frozenset represents a transaction.
        MIN_SUP: The minimum support threshold.
    Returns:
        A dictionary representing the frequent itemset table, where the keys are the itemsets
        and the values are dictionaries containing the following information:
          * count: The number of transactions in the dataset that contain the itemset.
          * support: The support of the itemset.
          * parent: The parent itemset in the table, or `None` if the itemset is the
            root of the tree.
    """
    itemset_counts = count_itemset_occurrences(D, generate_subsets)
    total_transactions = len(D)
    itemset_table = {}

    try:
        for itemset, count in itemset_counts.items():
            support = calculate_support(count, total_transactions)

            # If the support is greater than or equal to the minimum support threshold,
            # add the itemset to the itemset table.
            if support >= MIN_SUP:
                itemset_table[itemset] = {"count": count, "support": support, "parent": None}

        # Calculate the parent of each itemset in the itemset table.
        for itemset in itemset_table:
            set_parent_for_itemset(itemset, itemset_table, generate_subsets)
    except Exception as e:
        logging.error(f"An error occurred while creating the itemset table: {str(e)}")

    return itemset_table

def display_itemset_table_using_pandas(itemset_table: Dict[frozenset, Dict[str, Any]]):
    """Prints an itemset table in a human-readable format using the pandas library.
    Args:
        itemset_table: A dictionary representing the itemset table, where the keys are
        the itemsets and the values are dictionaries containing the following
        information:
          * count: The number of transactions in the dataset that contain the itemset.
          * support: The support of the itemset.
          * parent: The parent itemset in the itemset table, or `None` if the itemset is the
            root of the table.
    """
    # Filter out the empty itemset
    filtered_itemset_table = {itemset: info for itemset, info in itemset_table.items() if itemset}

    # Create a DataFrame from the filtered itemset table
    df = pd.DataFrame(filtered_itemset_table)

    # Display the DataFrame
    print("Itemset Table:")
    print(df.to_string())
    print()

def get_max_itemset(itemset_table: Dict[frozenset, Dict[str, Any]]) -> List[frozenset]:
    # Get the maximum itemset length
    max_length = max(len(itemset) for itemset in itemset_table)

    # Get the itemsets with the maximum length
    max_itemsets = [itemset for itemset in itemset_table if len(itemset) == max_length]

    return max_itemsets

def get_nonempty_subsets(itemset: frozenset) -> List[frozenset]:
    subsets = generate_subsets(itemset)
    nonempty_subsets = [subset for subset in subsets if subset != itemset and len(subset) > 0]
    return nonempty_subsets

def find_association_rules(itemset: frozenset, itemset_table: Dict[frozenset, Dict[str, Any]]) -> List[Tuple[frozenset, frozenset]]:
    """
    Finds association rules for a given itemset based on the provided itemset table.
    Args:
        itemset (frozenset): The itemset for which association rules are to be found.
        itemset_table (dict): The itemset table.
    Returns:
        list: A list of association rules, each represented as a tuple (A, B), where A and B
              are itemsets forming the association rule.
    """
    subsets = get_nonempty_subsets(itemset)
    association_rules = []

    try:
        for A in subsets:
            B = itemset - A

            # Check if both A and B are in the itemset table
            if A in itemset_table and B in itemset_table:
                association_rules.append((A, B))
            elif B in itemset_table:
                association_rules.append((B, A))
    except Exception as e:
        logging.error(f"An error occurred while finding association rules for itemset {itemset}: {str(e)}")

    return association_rules

def calculate_confidence(rule: Tuple[frozenset, frozenset], itemset_table: Dict[frozenset, Dict[str, Any]]) -> float:
    """Calculates the confidence for an association rule.
    Args:
        rule: A tuple representing the association rule (A, B).
        itemset_table: A dictionary representing the itemset table.
    Returns:
        The confidence of the association rule.
    """
    A, B = rule
    try:
        count_A = itemset_table[A]["count"]
        count_A_and_B = itemset_table[A.union(B)]["count"]

        if count_A != 0:
            confidence = count_A_and_B / count_A
            return confidence
        else:
            logging.warning(f"Division by zero for rule: {rule}")
            return -1  # Indicate a division by zero

    except KeyError as e:
        logging.error(f"KeyError: {str(e)}")
        return None

    except ZeroDivisionError as e:
        logging.error(f"ZeroDivisionError: {str(e)}")
        return None
    
def display_association_rules(itemset_table: Dict[frozenset, Dict[str, Any]], max_itemsets: List[frozenset], MIN_CONF: float) -> None:
    """
    Display association rules based on the provided itemset_table, max_itemsets,
    and minimum confidence.
    Args:
        itemset_table (dict): The itemset table.
        max_itemsets (list): List of last itemsets.
        MIN_CONF (float): The minimum confidence threshold.
    Returns:
        None
    """
    for itemset in max_itemsets:
        print(f"Association rules for itemset {itemset}:")
        association_rules = find_association_rules(itemset, itemset_table)
        for A, B in association_rules:
            print(f"{A} -> {B}")

    print("\nAssociation rules equal or over the MIN_CONF:")

    for itemset in max_itemsets:
        association_rules = find_association_rules(itemset, itemset_table)
        for A, B in association_rules:
            confidence = calculate_confidence((A, B), itemset_table)
            try:
                if confidence is not None and confidence >= MIN_CONF:
                    print(f"{A} -> {B}, Confidence: {confidence:.2%}")
            except TypeError as e:
                logging.error(f"TypeError: {str(e)}")


if __name__ == "__main__":
    D = [
        {"M", "O", "N", "K", "E", "Y"},
        {"D", "O", "N", "K", "E", "Y"},
        {"M", "A", "K", "E"},
        {"M", "U", "C", "K", "Y"},
        {"C", "O", "O", "K", "I", "E"}
    ]

    itemset_table = create_itemset_table(D, MIN_SUP)

    # Print the itemset table
    display_itemset_table_using_pandas(itemset_table)
    max_itemsets = get_max_itemset(itemset_table)

    print("Maximum length itemsets:")
    for itemset in max_itemsets:
        print(itemset)
        print()

        nonempty_subsets = get_nonempty_subsets(itemset)
        print(f"Nonempty subsets of {itemset}: {nonempty_subsets}")
        print()

        display_association_rules(itemset_table, max_itemsets, MIN_CONF)
