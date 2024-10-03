import os
import random
import pickle
import numpy as np
import networkx as nx
import openpyxl


RUN = 100
random.seed(2024)

workbook = openpyxl.load_workbook('networks/networks_list.xlsx')
worksheet = workbook.active
for row in worksheet.iter_rows(min_row=2,values_only=True):
    dataset = row[0]
    if dataset.isdigit():
        infile = open('networks/Benchmark/OLP_updated.pickle', 'rb')
        df = pickle.load(infile)
        df_edgelists = df['edges_id']
        edge_index = df_edgelists.iloc[int(dataset)]
        network = 'Benchmark_{}'.format(dataset)

        G = nx.Graph()
        G.add_edges_from(edge_index)
    else:
        network=dataset
        G = nx.read_edgelist('networks/{}.txt'.format(dataset), create_using=nx.Graph)
    print(network)
    statistic_path= 'networks/statistics/{0}/statistic/'.format(network)
    if not os.path.exists(statistic_path):
        os.makedirs(statistic_path)
    A=nx.to_numpy_array(G)
    mask = np.ones_like(A, dtype=bool)
    mask[np.tril_indices(A.shape[0])] = False
    edge_index = np.column_stack(np.where((A == 1) & mask))
    np.save(statistic_path+'{0}_edge_index.npy'.format(network), edge_index)

    for temp_run in range(RUN):
        shuffle_file=statistic_path+'{0}_shuffle_{1}.npy'.format(network,temp_run)
        np.random.shuffle(edge_index)
        np.save(shuffle_file, edge_index)