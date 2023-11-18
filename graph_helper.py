import networkx as nx
import itertools

def generate_all_graphs(num_nodes):
    # Generate all possible edges for the graph
    edges = list(itertools.product(range(num_nodes), range(num_nodes)))
    
    #remove self-edges
    for edge in edges:
        if edge[0] == edge[1]:
            edges.remove(edge)

    all_graphs = []

    for edge_combination in itertools.product([0, 1], repeat=num_nodes*(num_nodes-1)):
        # Create a new directed graph
        G = nx.DiGraph()

        # Add nodes to the graph
        G.add_nodes_from(range(num_nodes))

        # Add edges to the graph
        for (i, j), edge in zip(edges, edge_combination):
            if edge:
                G.add_edge(i, j)

        # Print the graph
        all_graphs.append(G)
    
    return all_graphs

def remove_dual_edges(graphs):
    all_graphs = []

    for graph in graphs:
        edges = graph.edges()
        traversed_edges = []
        remove = False
        
        for edge in edges:
            edge_set = set(edge)
            if edge_set in traversed_edges:
                remove = True   
            else:
                traversed_edges.append(edge_set)
        
        if not remove:
            all_graphs.append(graph)

    return all_graphs

def is_unique_to_set(graph_to_check, unique_graphs):
    for graph in unique_graphs:
        if nx.is_isomorphic(graph_to_check, graph):
            return False
    return True
    
def remove_isomorphic(graphs):
    unique_graphs = []
    for graph in graphs:
        if is_unique_to_set(graph, unique_graphs):
            unique_graphs.append(graph)
    
    return unique_graphs

def remove_cyclic(graphs):
    all_graphs = []
    for graph in graphs:
        if nx.is_directed_acyclic_graph(graph):
            all_graphs.append(graph)

    return all_graphs

def generate_graphs(num_nodes, verbose=False):
    #Generates all possible Directed, Non-self-loops, graphs 
    all_graphs = generate_all_graphs(num_nodes)

    #Filter out double graphs 
    all_graphs = remove_dual_edges(all_graphs)

    #Remove Isomorphic Graphs 
    all_graphs = remove_isomorphic(all_graphs)

    #Remove cyclic graphs
    all_graphs = remove_cyclic(all_graphs)

    if verbose:
        print("There are", len(all_graphs), "graphs. They are")

        #Print current graphs
        for graph in all_graphs:
            print(graph.nodes(), graph.edges())
    
    return all_graphs