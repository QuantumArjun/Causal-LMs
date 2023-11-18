import networkx as nx
import numpy as np

all_inference_templates = {
    2: [[0,0],
        [0,1],
        [1,0],
        [1,1]],

    3: [[0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [0,1,1],
        [1,0,1],
        [1,1,1]],

    4: [[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],
        [1,1,1,0],
        [0,1,1,1],
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,1],
        [1,0,1,1],
        [0,1,1,1],
        [1,0,0,1],
        [1,1,1,1]]
}

def propagate_helper(graph, end_result, successor):
    end_result[successor] = 1
    successors = list(graph.successors(successor))
    for successor in successors:
        end_result = propagate_helper(graph, end_result, successor)
    return end_result

def propagate(graph, inference):
    end_result = inference.copy()
    for i, item in enumerate(inference):
        if item == 1:
            successors = list(graph.successors(i))
            for successor in successors:
                end_result = propagate_helper(graph, end_result, successor)
    return end_result

def get_inferences(graph, verbose=False):
    num_nodes = graph.number_of_nodes()
    inference_template = all_inference_templates[num_nodes]
    observations = []
    for inference in inference_template:
        observation = propagate(graph, inference)
        observations.append((tuple(inference), tuple(observation)))
    return observations


