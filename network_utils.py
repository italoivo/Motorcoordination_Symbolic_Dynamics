import matplotlib.patches as mpatches
from collections import Counter
import networkx as nx
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def create_transition_network(ordered_states):
    """
    Create a weighted and directed network from an ordered list of states, excluding self-loops.
    The weights of the edges represent transition probabilities, and node weights represent 
    the fraction of time each node appears in the state sequence.
    """
    
    # Count transitions, excluding self-loops
    transitions = [(ordered_states[i], ordered_states[i+1]) for i in range(len(ordered_states)-1)]
    transition_counts = Counter(transitions)
    
    # Calculate total transitions from each state to get probabilities, excluding self-loops
    total_transitions_from_state = Counter([state[0] for state in transitions])
    
    # Count the frequency of each state in the ordered sequence
    state_frequency = Counter(ordered_states)
    total_states = len(ordered_states)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with weights as the fraction of time they appear
    for state, count in state_frequency.items():
        node_weight = count / total_states
        G.add_node(state, weight=node_weight)
    
    # Add edges with weights as transition probabilities
    for (state_i, state_j), count in transition_counts.items():
        probability = count / total_transitions_from_state[state_i]
        dist = -np.log(probability)
        if np.abs(dist) < 1e-8:
            dist = 0.0
        G.add_edge(state_i, state_j, weight=probability, distance=dist)
    
    return G

def create_transition_network_no_loops(ordered_states):
    """
    Create a weighted and directed network from an ordered list of states, excluding self-loops.
    The weights of the edges represent transition probabilities, and node weights represent 
    the fraction of time each node appears in the state sequence.
    """
    from collections import Counter
    import networkx as nx
    
    # Count transitions, excluding self-loops
    transitions = [(ordered_states[i], ordered_states[i+1]) for i in range(len(ordered_states)-1) if ordered_states[i] != ordered_states[i+1]]
    transition_counts = Counter(transitions)
    
    # Calculate total transitions from each state to get probabilities, excluding self-loops
    total_transitions_from_state = Counter([state[0] for state in transitions])
    
    # Count the frequency of each state in the ordered sequence
    state_frequency = Counter(ordered_states)
    total_states = len(ordered_states)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with weights as the fraction of time they appear
    for state, count in state_frequency.items():
        node_weight = count / total_states
        G.add_node(state, weight=node_weight, label=f"{node_weight:.2f}")
    
    # Add edges with weights as transition probabilities
    for (state_i, state_j), count in transition_counts.items():
        probability = count / total_transitions_from_state[state_i]
        dist = -np.log(probability)
        if np.abs(dist) < 1e-8:
            dist = 0.0
        G.add_edge(state_i, state_j, weight=probability, distance=dist, label=f"{probability:.2f}")
    
    return G

def visualize_network(G,position = None):
    """
    Visualize the directed network with edge labels showing transition probabilities,
    with adjustments for a larger plot and more spread-out nodes.
    """
    plt.figure(figsize=(16, 12))  # Make the plot larger

    # Position nodes using the spring layout with additional spacing parameters
    # The 'k' parameter controls the distance between the nodes and can be adjusted for more spread
    #pos = nx.spring_layout(G,k=1.5, iterations=50)  # Increase 'k' for more spread
    if position == None:
        pos = nx.spring_layout(G)
    else:
        pos = position

    edge_labels = nx.get_edge_attributes(G, 'label')

    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, font_size=8, font_weight='bold', arrowstyle='->', arrowsize=20)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.show()
    return

def calculate_symbol_condition(symbols_sequence, ordered_conditions, condition_ranges):
    """
    Calculate the probability of each condition for each symbol, based on the sequence of symbols and
    ranges where conditions are applied.

    Parameters:
    - symbols_sequence: Sequence of symbols (can contain repeats) over which conditions are applied.
    - ordered_conditions: List indicating the order of conditions as they appear across the dataset.
    - condition_ranges: List of tuples with each tuple indicating the start and stop indices (inclusive) for each condition's application.

    Returns:
    - A dictionary with symbols as keys and another dictionary as values, where the inner dictionary
      has conditions as keys and their probabilities as values.
    """
    # Initialize data structures for counting occurrences
    condition_count_per_symbol = defaultdict(lambda: defaultdict(int))
    total_condition_occurrences_per_symbol = defaultdict(int)
    symbol_frequencies = defaultdict(int)

    # Process each condition range
    for condition, ranges in zip(ordered_conditions, condition_ranges):
        start_index, end_index = ranges
        # Apply conditions to the specified ranges in the symbols sequence
        for i in range(start_index, end_index + 1):
            symbol = symbols_sequence[i]
            condition_count_per_symbol[symbol][condition] += 1
            total_condition_occurrences_per_symbol[symbol] += 1
            symbol_frequencies[symbol] += 1

    # Calculate probabilities
    probabilities = {symbol: {} for symbol in symbols_sequence}
    for symbol in condition_count_per_symbol:
        for condition in condition_count_per_symbol[symbol]:
            probabilities[symbol][condition] = condition_count_per_symbol[symbol][condition] / total_condition_occurrences_per_symbol[symbol]

    return probabilities, symbol_frequencies

def plot_graph(graph, position, annotation,dwelling_time):
    probabilities = annotation

    # Define graph layout
    if position == None:
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = position

    # Create figure and axis for the overall plot with increased size
    fig, ax = plt.subplots(figsize=(20, 20))  # Adjust figure size as needed
    ax.set_axis_off()

    # Draw the network edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, width=1, arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')

    # Define consistent colors for conditions
    conditions = set(cond for probs in probabilities.values() for cond in probs)
    color_map = {cond: plt.cm.tab20(i) for i, cond in enumerate(conditions)}

    # Smaller pie charts size adjustment
    #pie_size = 0.02  # Adjust the size of pie charts as needed

    # Create a pie chart for each node without labels, with consistent colors and smaller size
    for node in graph.nodes:
        sizes = list(probabilities[node].values())

        pie_size = 0.001*dwelling_time[node]
        colors = [color_map[cond] for cond in probabilities[node].keys()]
        
        # Convert node position to figure space for placing the pie chart
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        pie_pos = trans2(trans(pos[node]))  # Convert node position to figure space
        
        # Adjust the pie chart size and position
        pie_ax = fig.add_axes([pie_pos[0]-pie_size/2, pie_pos[1]-pie_size/2, pie_size, pie_size], aspect=1)
        pie_ax.pie(sizes, colors=colors, radius=1)
        pie_ax.set_aspect('equal')
        pie_ax.set_axis_off()

    # Create legend entries
    legend_handles = [mpatches.Patch(color=color_map[cond], label=cond) for cond in conditions]

    # Display the legend
    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(10, 10), title="Conditions")
    plt.show()

def separate_graphs_by_condition(G, annotations, dwelling_times):
    """
    Generate separate graphs for each condition with adjusted node weights. Nodes not present for a specific
    condition (or having weight zero) are removed from the corresponding graph.

    Parameters:
    - G: NetworkX graph, the original graph from create_transition_network
    - annotations: dict, keys are node ids, values are dicts with keys as conditions and values as weights
    - dwelling_times: dict, keys are node ids, values are dwelling times for those nodes

    Returns:
    - dict of graphs, keys are conditions, values are graphs with adjusted node weights
    """
    # Initialize the dictionary to hold a graph for each condition
    graphs_by_condition = {}

    # Collect all possible conditions across all nodes
    all_conditions = set(cond for ann in annotations.values() for cond in ann.keys())

    # Create a separate graph for each condition
    for condition in all_conditions:
        # Copy the original graph
        new_graph = copy.deepcopy(G)

        # Adjust node attributes or remove nodes based on specific condition availability
        for node in list(new_graph.nodes()):  # list to allow modification during iteration
            condition_weight = annotations.get(node, {}).get(condition, 0)
            if condition_weight == 0:
                # Remove the node if it does not have a weight for the current condition
                new_graph.remove_node(node)
            else:
                original_dwelling_time = dwelling_times[node]
                new_weight = condition_weight * original_dwelling_time
                new_graph.nodes[node]['weight'] = new_weight
        
        # Store the new graph under the condition
        graphs_by_condition[condition] = new_graph

    return graphs_by_condition

def update_positions_for_condition_graphs(graphs_by_condition, positions):
    """
    Update the position dictionary to match the nodes present in each condition-specific graph.

    Parameters:
    - graphs_by_condition: dict of NetworkX graphs, keys are conditions, values are graphs
    - positions: dict, keys are node ids, values are coordinates (e.g., tuples) of node positions

    Returns:
    - dict of dicts, keys are conditions, values are dicts of node positions
    """
    # Initialize the dictionary to hold positions for each condition graph
    positions_by_condition = {}

    # Update positions for each condition-specific graph
    for condition, graph in graphs_by_condition.items():
        # Filter the original positions to include only those nodes present in the current graph
        condition_positions = {node: positions[node] for node in graph.nodes() if node in positions}
        
        # Store the updated positions under the condition
        positions_by_condition[condition] = condition_positions

    return positions_by_condition

def plot_unannotated_graph(graph, position):
    # Define graph layout
    if position is None:
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = position

    # Create figure and axis for the overall plot with increased size
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_axis_off()

    # Extract weights and scale node sizes
    weights = [graph.nodes[node].get('weight', 1) for node in graph.nodes()]  # Default weight is 1 if not set
    max_weight = max(weights) if weights else 1  # Avoid division by zero
    node_sizes = [1000 * (weight / max_weight) for weight in weights]  # Scale factor of 1000 for visibility

    # Draw the network edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, width=1, arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.2', node_size=node_sizes)

    # Draw nodes with sizes proportional to weights
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='black', ax=ax)
    plt.show()
    return