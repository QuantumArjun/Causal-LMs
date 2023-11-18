import random

def sample_graphs(graphs, k=2):
    if k > len(graphs):
        assert "You're trying to sample more graphs than exist"
    else:
        return random.sample(graphs, k)

def sample_inferences(inference_dict, k, Verbose):

    observation_list = []
    for value in inference_dict.values():
        observation_list.append(value)

    # Get all lists
    sets = []
    for lst in [tuple(lst) for lst in observation_list]:
        sets.append(set(lst))

    # Find common pairs
    common_pairs = sets[0]
    for s in sets[1:]:
        common_pairs = common_pairs.intersection(s)

    # Find unique pairs in each list
    unique_pairs = []
    for i, s in enumerate(sets):
        unique_pairs.extend(s.difference(*sets[:i], *sets[i+1:]))

    unique_dict = inference_dict.copy()
    for key in unique_dict:
        for pair in common_pairs:
            if pair in unique_dict[key]:
                unique_dict[key].remove(pair)

    # Printing the results
    if Verbose:
        print("Common Pairs:", common_pairs)
        for key in unique_dict:
            print("The following observations are unique to", key, "\n", unique_dict[key])
    
    return (common_pairs, unique_dict)
