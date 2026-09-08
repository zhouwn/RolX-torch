# refex.py
import math
import random
import scipy.stats
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from functools import reduce
from par import parameter_parser


def dataset_reader(path):
    edges = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edges)
    # Ensure nodes are indexed from 0 to N-1
    return nx.convert_node_labels_to_integers(graph)


def inducer(graph, node):
    # In NetworkX 2.x, neighbors returns an iterator
    nebs = list(nx.neighbors(graph, node))
    sub_nodes = nebs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    # The map call was incorrect, fixed to correctly sum degrees
    out_counts = sum(d for n, d in graph.degree(sub_nodes))
    return sub_g, out_counts, nebs


def complex_aggregator(x):
    if x.size == 0:
        return [0.0] * 9  # Return default value for empty input
    return [np.min(x), np.std(x), np.var(x), np.mean(x), np.percentile(x, 25), np.percentile(x, 50),
            np.percentile(x, 75), scipy.stats.skew(x), scipy.stats.kurtosis(x)]


def aggregator(x):
    if x.size == 0:
        return [0.0, 0.0]  # Return default value for empty input
    return [np.sum(x), np.mean(x)]


def state_printer(x):
    print("-" * 80)
    print(x)
    print("")


def sub_selector(old_features, new_features, pruning_threshold):
    print("Cross-temporal feature pruning started.")
    indices_to_remove = set()
    for i in tqdm(range(old_features.shape[1])):
        for j in range(new_features.shape[1]):
            # Check for columns with zero variance to avoid correlation errors
            if np.std(old_features[:, i]) > 0 and np.std(new_features[:, j]) > 0:
                c = np.corrcoef(old_features[:, i], new_features[:, j])
                if abs(c[0, 1]) > pruning_threshold:
                    indices_to_remove.add(j)

    keep = list(set(range(new_features.shape[1])) - indices_to_remove)
    # Sort indices to maintain order
    keep.sort()
    return new_features[:, keep] if keep else np.zeros((new_features.shape[0], 0))


class RecursiveExtractor:
    def __init__(self, args):
        self.args = args
        self.aggregator = complex_aggregator if self.args.aggregator == "complex" else aggregator
        self.multiplier = len(self.aggregator(np.array([1, 2])))
        self.graph = dataset_reader(self.args.input)
        self.nodes = sorted(list(self.graph.nodes()))
        self.node_count = len(self.nodes)
        self.create_features()

    def basic_stat_extractor(self):
        self.base_features = []
        self.sub_graph_container = {}
        for node in tqdm(range(self.node_count)):
            sub_g, overall_counts, nebs = inducer(self.graph, node)
            in_counts = len(sub_g.edges())
            self.sub_graph_container[node] = nebs
            deg = sub_g.degree(node)
            trans = nx.clustering(self.graph, node)  # Clustering is better on the original graph

            # Handle division by zero
            ratio1 = float(in_counts) / float(overall_counts) if overall_counts > 0 else 0
            ratio2 = float(overall_counts - in_counts) / float(overall_counts) if overall_counts > 0 else 0

            self.base_features.append([in_counts, overall_counts, ratio1, ratio2, deg, trans])
        self.features = {0: np.array(self.base_features)}
        print("")
        del self.base_features

    def single_recursion(self, i):
        features_from_previous_round = self.features[i]
        feature_dim = features_from_previous_round.shape[1]
        new_features = np.zeros((self.node_count, feature_dim * self.multiplier))

        for k in tqdm(range(self.node_count)):
            selected_nodes = self.sub_graph_container[k]
            if not selected_nodes:
                continue  # Skip nodes with no neighbors

            main_features = features_from_previous_round[selected_nodes, :]

            aggregated = [self.aggregator(main_features[:, j]) for j in range(feature_dim)]
            new_features[k, :] = reduce(lambda x, y: x + y, aggregated)
        return new_features

    def do_recursions(self):
        for recursion in range(self.args.recursive_iterations):
            state_printer(f"Recursion round: {recursion + 1}.")

            # Skip recursion if previous round yielded no features
            if self.features[recursion].shape[1] == 0:
                self.features[recursion + 1] = np.zeros((self.node_count, 0))
                continue

            new_features = self.single_recursion(recursion)
            new_features = sub_selector(self.features[recursion], new_features, self.args.pruning_cutoff)
            self.features[recursion + 1] = new_features

        # Concatenate features from all rounds
        all_features = [self.features[k] for k in sorted(self.features.keys()) if self.features[k].shape[1] > 0]
        self.features = np.concatenate(all_features, axis=1)

        # Normalize features, handling division by zero
        min_val = np.min(self.features, axis=0)
        max_val = np.max(self.features, axis=0)
        range_val = max_val - min_val
        # Avoid division by zero for constant features
        range_val[range_val == 0] = 1.0
        self.features = (self.features - min_val) / range_val

    def binarize(self):
        self.new_features = []
        for x in tqdm(range(self.features.shape[1])):
            try:
                # Use pd.qcut to discretize and get_dummies for one-hot encoding
                binned_feature = pd.qcut(self.features[:, x], self.args.bins, labels=False, duplicates="drop")
                dummies = pd.get_dummies(binned_feature, prefix=f'feature_{x}')
                self.new_features.append(dummies)
            except Exception as e:
                # Silently pass if a feature cannot be binned (e.g., it's constant)
                pass
        self.new_features = pd.concat(self.new_features, axis=1)

    def dump_to_disk(self):
        self.new_features.columns = [f"x_{i}" for i in range(self.new_features.shape[1])]
        self.new_features.to_csv(self.args.recursive_features_output, index=None)

    def create_features(self):
        state_printer("Basic node level feature extraction and induced subgraph creation started.")
        self.basic_stat_extractor()
        state_printer("Recursion started.")
        self.do_recursions()

        if self.features.shape[1] == 0:
            state_printer("No features were generated after recursion and pruning.")
            # Create a dummy DataFrame if no features are left
            self.new_features = pd.DataFrame(index=range(self.node_count))
        else:
            state_printer("Binary feature quantization started.")
            self.binarize()
            state_printer("Saving the raw features.")
            self.dump_to_disk()

        state_printer(f"The number of extracted features is: {self.new_features.shape[1]}.")