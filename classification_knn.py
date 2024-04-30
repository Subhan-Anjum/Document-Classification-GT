import pandas as pd
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import docx
import tkinter as tk
from tkinter import filedialog
from nltk.stem import WordNetLemmatizer

# Function to preprocess text by tokenization, removing stopwords, and stemming



def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens if word.lower() not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return lemmatized_tokens


# Function to create a directed graph from text data


def create_graph_from_text_data(text):
    terms = preprocess_text(text)
    graph = nx.DiGraph()
    for i in range(len(terms)-1):
        graph.add_edge(terms[i], terms[i+1])
    return graph

# Function to classify a document using k-Nearest Neighbors (kNN) algorithm


def classify_document(test_graph, k):
    distances = []

    # Compute distance between the test graph and each training graph
    for train_id, train_graph in document_graphs.items():
        distance = compute_graph_distance(test_graph, train_graph)
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

# Function to compute distance between graphs based on their maximal common subgraph


def compute_graph_distance(graph1, graph2):
    mcs_graph = find_maximal_common_subgraph(graph1, graph2)
    return -len(mcs_graph.edges())

# Function to find the maximal common subgraph between two graphs


def find_maximal_common_subgraph(graph1, graph2):
    # Convert graphs to sets of edges
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())

    # Find common edges between the two graphs
    common_edges = edges1.intersection(edges2)

    # Create a new graph with the common edges
    mcs_graph = nx.Graph(list(common_edges))

    return mcs_graph

# Function to read the text content from a Word document


def read_word_document(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()


# Load data
data = pd.read_csv('output.csv', encoding='latin1')

# Preprocess training data and create graphs
content_column = 'Content'
document_graphs = {}
for index, row in data.iterrows():
    content = row[content_column]
    document_id = index
    document_graphs[document_id] = create_graph_from_text_data(content)

# Create GUI window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Prompt user to select a file
file_path = filedialog.askopenfilename(title="Select a .docx file")

# Read and preprocess the text content of the document
text_content = read_word_document(file_path)
test_graph = create_graph_from_text_data(text_content)

# Classify the document using kNN
predicted_category = classify_document(test_graph, k=3)
print("Predicted Class:", predicted_category)
