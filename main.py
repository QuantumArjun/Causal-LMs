import graph_helper
import inference_helper
import sample_helper
import networkx as nx
import argparse
from prompt_builder import PromptBuilder

def main(num_nodes, num_sample, test=False, verbose=False, type="base"):
    #Generated num_nodes number of Directed, Acyclical, Graphs 
    all_graphs = graph_helper.generate_graphs(num_nodes)

    print("There are", len(all_graphs), "graphs with", num_nodes, "nodes")

    #sample a certain number of graphs 
    print("We are sampling", num_sample, "graphs")
    sampled_graphs = sample_helper.sample_graphs(all_graphs, num_sample)
    if test:
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2)])
        F = nx.DiGraph()
        F.add_nodes_from([0, 1, 2])
        sampled_graphs = [G, F]
    for graph in sampled_graphs:
        print(graph.edges())

    #Generate the possible inferences for each graph 
    inference_dict = {}
    for graph in sampled_graphs:
        inference_dict[graph] = inference_helper.get_inferences(graph)

    all_inferences = {}

    if verbose:
        for key in inference_dict:
            print("Graph:", key, "Nodes", key.nodes(), "Edges", key.edges())
            output = [t[1] for t in inference_dict[key]]
            print("Input ", inference_helper.all_inference_templates[len(key.nodes())], "\nOutput", output)
            all_inferences[key] = (inference_helper.all_inference_templates[len(key.nodes())], output)

    common, unique_dict = sample_helper.sample_inferences(inference_dict, 2, Verbose=verbose)

    prompt_builder = PromptBuilder()
    prompt_builder.initialize(num_nodes, num_sample, common, unique_dict)
    prompt_builder.generate_prompt(sampled_graphs, all_inferences, unique_dict, common, type=type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Generation Arguments')
    parser.add_argument('--num_nodes', type=int, default=3)
    parser.add_argument('--num_sample', type=int, default=2)
    parser.add_argument('--type', type=str, default="base")
    args = parser.parse_args()

    main(args.num_nodes, args.num_sample, test=False, verbose=True, type=args.type)

    '''
    Fix observations 
    
    To-Do: Base Template - Done 

    To-Do: Counterfactual:
    - Write out the prompt, where you show two graphs, and show two actions, which action is enough to distinguish machines 
    - Ask what would have happened 

    To-Do: Intervention:
    1. Sample names for nodes randomly (nonsensical) - Using GPT
    2. Sample state names from a list of sensible actions (e.g. "open", "close", "on", "off", "not spin", "spin", "not push", "push")
    3. Store these in a csv
    4. Generate text from the csv (using a text template) - just the framing of the problem - Just the prompt. Num nodes, num states, related in some way
    5. Show the possible graphs
    6. Goal - you can toggle state (on or off), what action do you take
    7. Give it feedback, repeat until it guesses the correct graph


    ^ Options, extend base, give a graph and ask it what would've happened, give all possible graphs and and about the impact of an observation it hasn't seen 

    Later
    1. Choosing amongst 3 graphs
    '''
'''
Base prompt example

Prompt: There are 2 machines. All machines have 3 parts, a gase, a neke, and a sike. In a bape machine, when a gase is inflated, it causes neke to be inflated. In a bape machine, when a gase is inflated, it causes a sike to be inflated.

In a wube machine, when a neke is inflated, it causes a sike to be infalted. In a wube machine, when a gase is inflated, it causes a sike to be inflated.

Let's say that you have one of these machines, but don't know which one. The gase, neke, and sike are all deflated. Alice inflates the gase and the neke, and you see that the sike inflates. Which machine do you have?
*Update prompt to have 2 edges
'''

'''
Counterfactual prompt example
Prompt:
There are 2 machines. All machines have 3 parts, a gase, a neke, and a sike. 

In a bape machine, when a gase is inflated, it causes neke to be inflated. In a bape machine, when a gase is inflated, it causes a sike to be inflated.

In a wube machine, when a neke is inflated, it causes a sike to be infalted. In a wube machine, when a gase is inflated, it causes a sike to be inflated.

Let's say that you have one of these machines, but don't know which one. What parts would you inflate to determine which machine you have?
*Change prompt to include 2 graphs, and 2 actions, and ask which action is enough to distinguish the machines
'''

'''
Intervention prompt example
Prompt:
There are 2 machines. All machines have 3 parts, a gase, a neke, and a sike. In a bape machine, when a gase is inflated, it causes neke to be inflated. In a bape machine, when a gase is inflated, it causes a sike to be inflated.

In a wube machine, when a neke is inflated, it causes a sike to be infalted. In a wube machine, when a gase is inflated, it causes a sike to be inflated.

Let's say that you have one of these machines, but don't know which one. What machine do you have?

Note: You will need to try different actions in order to determine the blickets and the type of machine. Since you cannot see the machine, after you perform an action, wait for the user to give you the corresponding observation.

Available actions: [List of actions]*

Follow the following pattern to interact with the user:
Thought: [Your thoughts]
Action: [Your proposed action]
Observation: [You must wait for the user to reply with the obervation]
Parse the action, and then feed as the obervation 
'''