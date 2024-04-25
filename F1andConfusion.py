# import pandas as pd
# import networkx as nx
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from sklearn.metrics import f1_score, confusion_matrix
# from collections import Counter
# from preprocessing import document_graphs, create_graph

# # Download NLTK resources if needed
# # nltk.download('punkt')
# # nltk.download('stopwords')

# def preprocess_text(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     porter = PorterStemmer()
#     stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
#     return stemmed_tokens

# def create_graph(text):
#     terms = preprocess_text(text)
#     G = nx.DiGraph()
#     for i in range(len(terms)-1):
#         G.add_edge(terms[i], terms[i+1])
#     return G

# def knn_classify(test_graph, k):
#     distances = []

#     # Compute distance between test_graph and each training graph
#     for train_id, train_graph in document_graphs.items():
#         distance = compute_distance(test_graph, train_graph)
#         distances.append((train_id, distance))

#     # Sort distances in ascending order
#     distances.sort(key=lambda x: x[1])

#     # Get the k-nearest neighbors
#     neighbors = distances[:k]

#     # Get categories of the neighbors
#     neighbor_categories = [data.loc[i, 'Type'] for i, _ in neighbors]

#     # Find the majority class
#     majority_class = Counter(neighbor_categories).most_common(1)[0][0]

#     return majority_class

# def compute_distance(G1, G2):
#     mcs_graph = compute_mcs(G1, G2)
#     return -len(mcs_graph.edges())

# def compute_mcs(G1, G2):
#     # Convert graphs to edge sets
#     edges1 = set(G1.edges())
#     edges2 = set(G2.edges())

#     # Compute the intersection of edges
#     common_edges = edges1.intersection(edges2)

#     # Create a new graph with common edges
#     mcs_graph = nx.Graph(list(common_edges))

#     return mcs_graph

# # Initialize lists to store true and predicted labels
# y_true = []
# y_pred = []

# # True labels for the test documents
# true_labels = ['Travel','Travel','Travel', 'Fashion and Beauty','Fashion and Beauty','Fashion and Beauty','Diseases and Symptoms','Diseases and Symptoms','Diseases and Symptoms']  # Replace with your actual categories

# data = pd.read_csv('merged_file.csv', encoding='latin1')
# test_documents = [create_graph(str(data.iloc[12]['Content'])), create_graph(
#     str(data.iloc[13]['Content'])), create_graph(str(data.iloc[14]['Content'])),create_graph(str(data.iloc[27]['Content'])),create_graph(str(data.iloc[28]['Content'])),create_graph(str(data.iloc[29]['Content'])),create_graph(str(data.iloc[42]['Content'])),create_graph(str(data.iloc[43]['Content'])),create_graph(str(data.iloc[44]['Content']))]

# # Classify test documents using kNN
# for test_graph in test_documents:
#     predicted_category = knn_classify(test_graph, k=3)
#     # Add true label to y_true
#     y_true.append(true_labels[test_documents.index(test_graph)])
#     # Add predicted label to y_pred
#     y_pred.append(predicted_category)

# # Calculate F1 score
# f1 = f1_score(y_true, y_pred, average='weighted')

# # Calculate confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)
# labels = sorted(set(true_labels))

# # Calculate accuracy in percentage
# accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true) * 100

# # Print F1 score
# print("F1 Score:", f1)

# # Print accuracy
# print("Accuracy:", accuracy, "%")

# # Create DataFrame for confusion matrix
# confusion_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# # Add actual and predicted labels to the DataFrame
# confusion_df.index.name = 'Actual'
# confusion_df.columns.name = 'Predicted'

# # Print the confusion matrix with labels
# print("Confusion Matrix:")
# print(confusion_df)

import pandas as pd
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from collections import Counter
from preprocessing import document_graphs, create_graph
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]
    return stemmed_tokens

def create_graph(text):
    terms = preprocess_text(text)
    G = nx.DiGraph()
    for i in range(len(terms)-1):
        G.add_edge(terms[i], terms[i+1])
    return G

def knn_classify(test_graph, k):
    distances = []

    # Compute distance between test_graph and each training graph
    for train_id, train_graph in document_graphs.items():
        distance = compute_distance(test_graph, train_graph)
        distances.append((train_id, distance))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])

    # Get the k-nearest neighbors
    neighbors = distances[:k]

    # Get categories of the neighbors
    neighbor_categories = [data.loc[i, 'Type'] for i, _ in neighbors]

    # Find the majority class
    majority_class = Counter(neighbor_categories).most_common(1)[0][0]

    return majority_class

def compute_distance(G1, G2):
    mcs_graph = compute_mcs(G1, G2)
    return -len(mcs_graph.edges())

def compute_mcs(G1, G2):
    # Convert graphs to edge sets
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    # Compute the intersection of edges
    common_edges = edges1.intersection(edges2)

    # Create a new graph with common edges
    mcs_graph = nx.Graph(list(common_edges))

    return mcs_graph

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# True labels for the test documents
true_labels = ['Travel','Travel','Travel', 'Fashion and Beauty','Fashion and Beauty','Fashion and Beauty','Diseases and Symptoms','Diseases and Symptoms','Diseases and Symptoms']  # Replace with your actual categories

data = pd.read_csv('merged_file.csv', encoding='latin1')
test_documents = [create_graph(str(data.iloc[12]['Content'])), create_graph(
    str(data.iloc[13]['Content'])), create_graph(str(data.iloc[14]['Content'])),create_graph(str(data.iloc[27]['Content'])),create_graph(str(data.iloc[28]['Content'])),create_graph(str(data.iloc[29]['Content'])),create_graph(str(data.iloc[42]['Content'])),create_graph(str(data.iloc[43]['Content'])),create_graph(str(data.iloc[44]['Content']))]

# Classify test documents using kNN
for test_graph in test_documents:
    predicted_category = knn_classify(test_graph, k=3)
    # Add true label to y_true
    y_true.append(true_labels[test_documents.index(test_graph)])
    # Add predicted label to y_pred
    y_pred.append(predicted_category)

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average='weighted')

# Calculate accuracy in percentage
accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true) * 100

# Calculate precision
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_true, y_pred, average='weighted')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
labels = sorted(set(true_labels))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print F1 score, accuracy, precision, and recall
print("F1 Score:", f1)
print("Accuracy:", accuracy, "%")
print("Precision:", precision)
print("Recall:", recall)
