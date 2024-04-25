import networkx as nx
import matplotlib.pyplot as plt

# Function to visualize a graph from a .graphml file
def visualize_graph(graphml_file):
    # Load the graph from the .graphml file
    G = nx.read_graphml(graphml_file)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
    
    # Display the graph
    plt.title("Graph Visualization")
    
    
    plt.show()

# Example usage
if __name__ == "__main__":
    graphml_file = "./GraphsProcessedProcessedCTravel/Document 2.graphml"
    visualize_graph(graphml_file)
