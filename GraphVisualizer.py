import tkinter as tk
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
import docx
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Function to preprocess text and create a directed graph
def preprocess_text_and_create_graph(text):
    porter = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [porter.stem(word) for word in tokens]
    G = nx.DiGraph()
    for i in range(len(stemmed_tokens)-1):
        G.add_edge(stemmed_tokens[i], stemmed_tokens[i+1])
    return G

# Function to visualize a graph from a .graphml file or .docx file
def visualize_graph():
    # Create GUI window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt user to select a file
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("GraphML files", "*.graphml"), ("Word files", "*.docx")])

    # Load the graph from the selected file
    if file_path.endswith('.graphml'):
        G = nx.read_graphml(file_path)
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        G = preprocess_text_and_create_graph(text)
    else:
        print("Unsupported file format. Please select a .graphml or .docx file.")
        return

    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)

    # Display the graph
    plt.title("Graph Visualization")
    plt.show()

# Example usage
if __name__ == "__main__":
    visualize_graph()
