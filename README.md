# causal-lms

## Experimenting with the Causal Reasoning capabilities of Large Language Models

Currently, the code is able to generate all feasible graphs (where feasible is defined as directed, acyclic graphs where each graph is isomorphic to each other, and each node has at most in or out node). Then, it samples a user specified number of graphs, and generates all possible observations from that causal graph.

## Usage: 

Sample command

<code>python3 main.py --num_nodes 3 --num_sample 2 --type base</code>

args:
* --num_nodes - The number of nodes in the causal graph (numbers > 4 will take a lot of time). Default = 3
* --num_sample - How many causal graphs should be sampled. Default = 2
* --type - Which experiment to run. There are 3 types: base, counterfactual, and intervention. Default = base