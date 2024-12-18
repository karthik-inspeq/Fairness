import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math

import pandas as pd


def input_parser(threshold, data, attribute, previledged, unpreviledged, metric_name, percentage):
    """
    Parses input data and maps metric values to binary (0/1) based on the threshold.
    Dynamically filters based on the given attributes and privileged/unprivileged logic.

    Args:
        threshold: Threshold value for converting True_label and Pred_label into binary.
        data: Input pandas DataFrame.
        attribute: List of attributes to dynamically filter on (e.g., ['Gender', 'Race']).
        previledged: List of privileged group members.
        unpreviledged: List of unprivileged group members.
        metric_name: Metric name to filter rows.

    Returns:
        filtered_data: Filtered DataFrame after applying all rules.
    """

    # Copy the data to avoid changes to the original DataFrame
    updated_data = data.copy()
    updated_data = updated_data[updated_data["Metric"] == metric_name]
    # Map 'True_label' and 'Pred_label' to binary values based on threshold
    for label in ["True_label", "Pred_label"]:
        if label in updated_data.columns:
            updated_data[label] = updated_data[label].apply(lambda x: 1 if x > threshold else 0)
        else:
            raise KeyError(f"Column '{label}' is missing in the data")

    # Add binary group column, safely checking for valid inputs
    if "Group" in updated_data.columns:
        updated_data["Group_binary"] = updated_data["Group"].map(
            lambda x: 1 if x in previledged else 0 if x in unpreviledged else None
        )
    else:
        raise KeyError("Column 'Group' is missing in the data")

    # Drop rows where 'Group_binary' is None (no match found)
    updated_data = updated_data.dropna(subset=["Group_binary"])
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    # Randomly sample a percentage of the rows
    fraction = percentage / 100.0  # Convert percentage to a fraction
    updated_data = updated_data.sample(frac=fraction, random_state=42) 
    return updated_data

# Defining Bias index
def bias_index(filtered_data, attribute):
    """The bias index for each attribute which will be used to calculate fiarness score

    Args:
        filtered_data (dataframe): Filtered dataframe based on user inputs
        attribute (str): The attribute selected to calculate bias index

    Returns:
        dict: A dictionary with all the calculated values.
    """
    
    ped = calculate_ped(filtered_data, attribute)
    eod = calculate_eod(filtered_data, attribute)
    bias_index_score = math.sqrt((ped**2 + eod**2)/2)
    return ped, eod, bias_index_score
def fairness_score(filtered_data, attributes):
    fairness_dict = {}
    bias_scores_squared = []
    for attribute in attributes:
        ped, eod, bias_score = bias_index(filtered_data= filtered_data, attribute=attribute)
        fairness_dict[attribute] = {
            "PED": ped,
            "EOD": eod,
            "Bias_Index": bias_score
        }
        bias_scores_squared.append(bias_score**2)
    fairness_value = 1 - math.sqrt(sum(bias_scores_squared)/len(attributes))
    fairness_dict["Fairness_Score"]=fairness_value
    group_data = {key: value for key, value in fairness_dict.items() if isinstance(value, dict)}
    # Convert to a DataFrame
    final_df = pd.DataFrame.from_dict(group_data, orient='index')
    return final_df, fairness_value

from sklearn.metrics import confusion_matrix
import pandas as pd

def calculate_eod(data, group_label):
    """
    Equal Opportunity Difference (EOD): Difference in True Positive Rate (TPR) between groups.
    """
    assert "True_label" in data.columns and "Pred_label" in data.columns, "Required columns missing."
    
    true_label = "True_label"
    pred_label = "Pred_label"

    # Calculate TPR for both groups
    tpr_group_0 = calculate_tpr(data, true_label, pred_label, group_label, 0)
    tpr_group_1 = calculate_tpr(data, true_label, pred_label, group_label, 1)

    # Return absolute difference
    return abs(tpr_group_0 - tpr_group_1)

def calculate_ped(data, group_label):
    """
    Predictive Equality Difference (PED): Difference in False Positive Rate (FPR) between groups.
    """
    true_label = "True_label"
    pred_label = "Pred_label"

    # Calculate FPR for both groups
    fpr_group_0 = calculate_fpr(data, true_label, pred_label, group_label, 0)
    fpr_group_1 = calculate_fpr(data, true_label, pred_label, group_label, 1)

    # Return absolute difference
    return abs(fpr_group_0 - fpr_group_1)

def calculate_tpr(data, true_label, pred_label, group_label, group_value):
    """
    Calculate True Positive Rate (TPR) for a specific group.
    """
    group_attribute = data[data["Attribute"] == group_label]
    group_data = group_attribute[group_attribute["Group_binary"] == group_value]
    if group_data.empty:
        return 0  # Handle case where group is missing in data

    # Confusion matrix with explicit labels
    cm = confusion_matrix(group_data[true_label], group_data[pred_label], labels=[0, 1])

    # Extract TP and FN
    tp = cm[1, 1]
    fn = cm[1, 0]

    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_fpr(data, true_label, pred_label, group_label, group_value):
    """
    Calculate False Positive Rate (FPR) for a specific group.
    """
    group_attribute = data[data["Attribute"] == group_label]
    group_data = group_attribute[group_attribute["Group_binary"] == group_value]
    if group_data.empty:
        return 0  # Handle case where group is missing in data

    # Confusion matrix with explicit labels
    cm = confusion_matrix(group_data[true_label], group_data[pred_label], labels=[0, 1])

    # Extract FP and TN
    fp = cm[0, 1]
    tn = cm[0, 0]

    return fp / (fp + tn) if (fp + tn) > 0 else 0

# Example Usage
if __name__ == "__main__":
    # Example data
    data_dict = {
    "gender": ["Male", "Female", "Male"],
    "race": ["Black", "White", "Asian"],
    "Metric": ["TOXICITY", "TOXICITY", "ROUGE_SCORE"] ,
    "Metric_label": [0.7, 0.4, 0.6],
    "Score": [0.6, 0.8, 0.9]
}
    data = pd.DataFrame(data_dict)
    metric = "TOXICITY"
    attributes = ["gender", "race"]
    previledged = ["Male", "White"]
    unpreviledged = ["Female", "Black"]
    final_result = input_parser(threshold=0.5, data = data, metric_name = metric, attribute= attributes, previledged=previledged, unpreviledged=unpreviledged)
    # Calculate fairness metrics
    final_fairness = fairness_score(final_result, attributes=attributes)
    print(f"Equal Opportunity Difference (EOD): {eod:.4f}")
    print(f"Predictive Equality Difference (PED): {ped:.4f}")
    print(f"bias_score is {bias_score:.4f}")
    print(f"fairness dict {final_fairness}")