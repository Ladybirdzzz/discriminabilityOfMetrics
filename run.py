import os
import random
import numpy as np
import networkx as nx
import openpyxl

import link_prediction_algorithms as lp
from metrics import prediction_evaluation

RUN = 100
test_p = 0.1
random.seed(2024)

algorithms = ['CN', 'RA', 'JA', 'PA', 'CH2', 'CN3', 'RA3', 'CH3', 'LRW', 'SRW', 'KA', 'MFI', 'SR', 'NMF', 'DW', 'N2V','GCN', 'GAT', 'SAGE', 'VGNAE']

workbook = openpyxl.load_workbook('networks/networks_list.xlsx')
worksheet = workbook.active
for row in worksheet.iter_rows(min_row=2, values_only=True):
    network = row[0]
    if network.isdigit():
        network = 'Benchmark_{}'.format(network)
    statistic_path = 'networks/statistics/{0}/statistic/'.format(network)
    edge_index=np.load(statistic_path + '{0}_edge_index.npy'.format(network))

    path = 'result/{0}/'.format(network)
    file_dict = {}
    for alg in algorithms:
        if not os.path.exists(path + alg):
            os.makedirs(path + alg)
        file_path = 'result/{0}/{1}/details.txt'.format(network, alg)
        file_object = open(file_path, mode='w')
        file_dict[alg] = file_object

    node_num = np.amax(edge_index) + 1
    edge_num = len(edge_index)
    test_num = int(edge_num * test_p)

    A = np.zeros((node_num, node_num), dtype=int)
    for [x, y] in edge_index:
        A[x][y] = 1
        A[y][x] = 1
    mask = np.ones_like(A, dtype=bool)
    mask[np.tril_indices(A.shape[0])] = False
    zero_index = np.column_stack(np.where((A == 0) & mask))

    y_label = np.concatenate((np.ones(test_num, dtype=int), np.zeros(zero_index.shape[0], dtype=int)), axis=0)

    for temp_run in range(RUN):
        shuffle_file=statistic_path+'{0}_shuffle_{1}.npy'.format(network,temp_run)
        edge_index = np.load(shuffle_file)

        test_mask = np.concatenate((np.ones(test_num, dtype=bool), np.zeros(edge_num - test_num, dtype=bool)), axis=0)
        pred_index = np.concatenate((edge_index[test_mask], zero_index), axis=0)
        for train_p in np.linspace(0, 1, 11):
            train_p = np.round(train_p, 1)
            print(network, temp_run,train_p)
            train_num = int(edge_num * (1 - test_p) * train_p)
            train_mask = np.concatenate((np.zeros(test_num, dtype=bool), np.ones(train_num, dtype=bool),
                                         np.zeros(edge_num - test_num - train_num, dtype=bool)), axis=0)
            G = nx.Graph()
            for node in range(node_num):
                G.add_node(node)
            G.add_edges_from(edge_index[train_mask])
            train_data = nx.to_numpy_array(G)
            sim_dict = {}
            sim_dict['CN'] = lp.Common_Neighbors(train_data, pred_index)
            sim_dict['RA'] = lp.Resource_Allocation(train_data, pred_index)
            sim_dict['JA'] = lp.Jaccard(train_data, pred_index)
            sim_dict['PA'] = lp.Preferential_Attachment(train_data, pred_index)
            sim_dict['CH2'], sim_dict['CH3'], sim_dict['CN3'], sim_dict['RA3'] = lp.Cannistraci_Hebb_L2_L3(train_data, pred_index)
            sim_dict['LRW'] = lp.Local_Random_Walk(train_data, pred_index)
            sim_dict['SRW'] = lp.Superposed_Random_Walk(train_data, pred_index)
            sim_dict['KA'] = lp.Katz(train_data, pred_index)
            sim_dict['MFI'] = lp.Matrix_Forest_Index(train_data, pred_index)
            sim_dict['SR'] = lp.simRank(G, pred_index)
            sim_dict['NMF'] = lp.sim_NetMF(G, pred_index)
            sim_dict['DW'] = lp.sim_DeepWalk(G, pred_index)
            sim_dict['N2V'] = lp.sim_Node2Vec(G, pred_index)
            sim_dict['GCN'] = lp.sim_GNN('GCN',node_num, train_num, edge_index, test_mask, train_mask, zero_index)
            sim_dict['GAT'] = lp.sim_GNN('GAT', node_num, train_num, edge_index, test_mask, train_mask, zero_index)
            sim_dict['SAGE'] = lp.sim_GNN('SAGE', node_num, train_num, edge_index, test_mask, train_mask, zero_index)
            sim_dict['VGNAE'] = lp.sim_VGNAE('VGNAE', node_num, train_num, edge_index, test_mask, train_mask, zero_index)
            for alg in algorithms:
                score = lp.nor_sim(sim_dict[alg])
                prec, auc_prec, auc_pr, auc_roc, auc_mroc, ndcg, mcc, h_measure = prediction_evaluation(y_label, score)
                file_dict[alg].write(str(temp_run) + ' ' + str(train_p) + ' ' + str(prec) + ' ' + str(auc_prec) + ' ' + str(
                    auc_pr) + ' ' + str(auc_roc) + ' ' + str(auc_mroc) + ' ' + str(ndcg) + ' ' + str(mcc) + ' ' + str(
                    h_measure) + '\n')

    for alg in algorithms:
        file_dict[alg].close()


