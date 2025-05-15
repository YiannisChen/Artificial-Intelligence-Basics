import pandas as pd
import math
from collections import Counter


# Calculate entropy
def calcEntropy(dataSet):
    total = len(dataSet)
    label_counts = Counter([data[-1] for data in dataSet])  # Count each class label
    entropy = 0.0
    for label, count in label_counts.items():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


# Calculate Information Gain
def calcInfoGain(dataSet, featureIndex):
    totalEntropy = calcEntropy(dataSet)

    # Split the dataset by the unique values of the chosen feature
    feature_values = set([data[featureIndex] for data in dataSet])

    weighted_entropy = 0.0
    for value in feature_values:
        subset = [data for data in dataSet if data[featureIndex] == value]
        prob = len(subset) / len(dataSet)
        weighted_entropy += prob * calcEntropy(subset)

    # Information Gain
    info_gain = totalEntropy - weighted_entropy
    return info_gain


# Calculate Information Gain Ratio (C4.5)
def calcInfoGainRatio(dataSet, featureIndex):
    totalEntropy = calcEntropy(dataSet)

    # Split the dataset by the unique values of the chosen feature
    feature_values = set([data[featureIndex] for data in dataSet])

    weighted_entropy = 0.0
    for value in feature_values:
        subset = [data for data in dataSet if data[featureIndex] == value]
        prob = len(subset) / len(dataSet)
        weighted_entropy += prob * calcEntropy(subset)

    # Information Gain
    info_gain = totalEntropy - weighted_entropy

    # Calculate the intrinsic information
    intrinsic_info = 0.0
    for value in feature_values:
        prob = len([data for data in dataSet if data[featureIndex] == value]) / len(dataSet)
        intrinsic_info -= prob * math.log2(prob)

    if intrinsic_info == 0:  # Avoid division by zero
        return 0
    gain_ratio = info_gain / intrinsic_info
    return gain_ratio



if __name__ == "__main__":
    # Sample dataset (watermelon features)
    dataset = [
        ['sunny', 'hot', 'high', 'weak', 'no'],
        ['sunny', 'hot', 'high', 'strong', 'no'],
        ['overcast', 'hot', 'high', 'weak', 'yes'],
        ['rain', 'mild', 'high', 'weak', 'yes'],
        ['rain', 'cool', 'normal', 'weak', 'yes'],
        ['rain', 'cool', 'normal', 'strong', 'no'],
        ['overcast', 'cool', 'normal', 'strong', 'yes'],
        ['sunny', 'mild', 'high', 'weak', 'no'],
        ['sunny', 'cool', 'normal', 'weak', 'yes'],
        ['rain', 'mild', 'normal', 'weak', 'yes'],
        ['sunny', 'mild', 'normal', 'strong', 'yes'],
        ['overcast', 'mild', 'high', 'strong', 'yes'],
        ['overcast', 'hot', 'normal', 'weak', 'yes'],
        ['rain', 'mild', 'high', 'strong', 'no']
    ]

    # Calculate information gain for the first feature (index 0)
    feature_index = 0
    info_gain = calcInfoGain(dataset, feature_index)
    print(f"Information Gain for feature {feature_index}: {info_gain}")

    # Calculate information gain ratio for the first feature (index 0)
    info_gain_ratio = calcInfoGainRatio(dataset, feature_index)
    print(f"Information Gain Ratio for feature {feature_index}: {info_gain_ratio}")
