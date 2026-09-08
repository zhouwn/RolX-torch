import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook tvshow network.
    The default hyperparameters give a good quality representation and good candidate cluster means without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run RolX.")

    #------------------------------------------------------------------
    # Input and output file parameters.
    #------------------------------------------------------------------

    parser.add_argument("--input", nargs = "?", default = r"D:\experiment\network dismantling\RolX-torch\input\Douban.csv", help = "Input graph path.")

    parser.add_argument("--recursive-features-output", nargs = "?", default = r"D:\experiment\network dismantling\RolX-torch\output\features\Douban_features.csv", help = "Embeddings path.")

    parser.add_argument("--embedding-output", nargs = "?", default = r"D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_embedding.csv", help = "Embeddings path.")

    parser.add_argument("--log-output", nargs = "?", default = r"D:\experiment\network dismantling\RolX-torch\output\logs\Douban_log.json", help = "Log path.")

    #-----------------------------------------------------------------------
    # Recursive feature extraction parameters.
    #-----------------------------------------------------------------------

    parser.add_argument("--recursive-iterations", type = int, default = 3, help = "Number of recursions.")

    parser.add_argument("--aggregator", nargs = "?", default = "simple", help = "Aggregator statistics extracted.")

    parser.add_argument("--bins", type = int, default = 40, help = "Number of quantization bins.")

    parser.add_argument("--pruning-cutoff", type = float, default = 0.5, help = "Absolute correlation for feature pruning.")

    #------------------------------------------------------------------
    # Factor model parameters.
    #------------------------------------------------------------------

    parser.add_argument("--dimensions", type = int, default = 1000, help = "Number of dimensions. Default is 16.")

    parser.add_argument("--batch-size", type = int, default = 320, help = "Number of edges in batch. Default is 128.")

    parser.add_argument("--epochs", type = int, default = 5, help = "Number of epochs. Default is 50.")

    parser.add_argument("--initial-learning-rate", type = float, default = 0.01, help = "Initial learning rate. Default is 0.01.")

    parser.add_argument("--minimal-learning-rate", type = float, default = 0.001, help = "Minimal learning rate. Default is 0.001.")

    parser.add_argument("--annealing-factor", type = float, default = 1, help = "Annealing factor. Default is 1.0.")

    parser.add_argument("--stage-dim", type=int, default=512, help="每一阶段训练的维度。")

    return parser.parse_args()
