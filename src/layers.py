# layers.py
import torch
import torch.nn as nn


class Factorization(nn.Module):
    """
    Factorization layer class using PyTorch.
    This model learns a node embedding and a feature embedding
    to reconstruct the node-feature matrix.
    """

    def __init__(self, args, user_size, feature_size):
        """
        Initialization of the layer with PyTorch embedding layers.
        """
        super(Factorization, self).__init__()
        self.args = args
        self.user_size = user_size
        self.feature_size = feature_size

        # Node embedding layer
        self.embedding_node = nn.Embedding(self.user_size, self.args.dimensions)

        # Feature embedding layer
        self.embedding_feature = nn.Embedding(self.feature_size, self.args.dimensions)

        # Initialize weights similar to the original TensorFlow implementation
        init_range = 0.1 / self.args.dimensions
        nn.init.uniform_(self.embedding_node.weight.data, -init_range, init_range)
        nn.init.uniform_(self.embedding_feature.weight.data, -init_range, init_range)

    def forward(self, edge_indices_left, edge_indices_right):
        """
        Performs the forward pass to predict the node-feature matrix.
        Args:
            edge_indices_left: Tensor with node indices.
            edge_indices_right: Tensor with feature indices.
        Returns:
            Predicted node-feature interactions.
        """
        # Look up embeddings for the given indices
        embedding_left = self.embedding_node(edge_indices_left)
        embedding_right = self.embedding_feature(edge_indices_right)

        # Predict interactions via matrix multiplication and apply sigmoid
        # (batch_size, D) @ (D, feature_size) -> (batch_size, feature_size)
        predictions = torch.sigmoid(torch.matmul(embedding_left, embedding_right.t()))

        return predictions