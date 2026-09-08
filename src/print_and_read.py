# print_and_read.py
import json
import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable

def data_reader(input_path):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    """    
    data = pd.read_csv(input_path).values
    return data

def log_setup(args_in):
    """
    Function to setup the logging hash table.
    """    
    log = dict()
    log["times"] = []
    log["losses"] = []
    log["new_features_added"] = []
    log["params"] = vars(args_in)
    return log

def tab_printer(log):
    """
    Function to print the logs in a nice tabular format.
    """    
    t = Texttable() 
    # Check if losses exist to prevent error on first call
    if log["losses"]:
        t.add_rows([["Epoch", log["losses"][-1][0]]])
        print(t.draw())

        t = Texttable()
        t.add_rows([["Loss", round(log["losses"][-1][1], 5)]])
        print(t.draw())    

def epoch_printer(repetition):
    """
    Function to print the epoch number.
    """    
    print("\n" + "="*20)
    print(f"Epoch {repetition+1} initiated.")
    print("="*20 + "\n")

def log_updater(log, repetition, average_loss, optimization_time):
    """ 
    Function to update the log object.
    """    
    index = repetition + 1
    log["losses"].append([index, average_loss])
    log["times"].append([index, optimization_time])
    return log

def data_saver(features, place):
    """
    Function to save the features (embeddings) to a CSV file.
    """
    columns = [f"x_{i}" for i in range(features.shape[1])]
    features = pd.DataFrame(features, columns=columns)
    features.to_csv(place, index=None)