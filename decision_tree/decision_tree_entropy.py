import pandas as pd
import numpy as np
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz


# calculate entropy
def calculate_entropy(dataset, label_col):
    label_counts = dataset[label_col].value_counts()
    total = len(dataset)
    entropy = -sum((count / total) * np.log2(count / total) for count in label_counts)
    return entropy


if __name__ == '__main__':
    # Load dataset
    column_names = ['age', 'prescript', 'astigmatic', 'tearRate', 'class']
    data = pd.read_csv('lenses.txt', header=None, names=column_names, sep='\t')
    print("Data Head:\n", data.head())

    # Calculate entropy
    entropy_value = calculate_entropy(data, 'class')
    print(f"Entropy of the dataset (based on 'class' column): {entropy_value}")

    # Separate features
    features = data.drop('class', axis=1)
    target = data['class']

    # Initialize LabelEncoders
    feature_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    # Encode features
    for col in features.columns:
        features[col] = feature_encoder.fit_transform(features[col])

    # Encode target
    target_encoded = target_encoder.fit_transform(target)

    # Display data
    print("Encoded Features:\n", features.head())
    print("Encoded Target:\n", target_encoded[:5])

    # Split data into features and target
    X = features
    y = target_encoded

    # Train-test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train
    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate and print the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Decision Tree Classifier: {accuracy * 100:.2f}%")

    # visualization
    dot_data = export_graphviz(
        classifier,
        out_file=None,
        feature_names=X.columns,
        class_names=target_encoder.classes_,
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Create graph
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Save the decision tree visualization as a PNG file
    graph.write_png("decision_tree.png")
    print("Decision tree saved as 'decision_tree.png'.")
