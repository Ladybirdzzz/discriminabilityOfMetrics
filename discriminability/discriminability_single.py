import numpy as np
from openpyxl import Workbook
import pandas as pd

df = pd.read_excel('../networks/networks_list.xlsx')
datasets = df.values
metrics = ['prec', 'auc_prec', 'auc_pr', 'auc_roc', 'auc_mroc', 'ndcg', 'mcc', 'h_measure']
lps = ['CN', 'RA', 'JA', 'PA', 'CH2', 'CN3', 'RA3', 'CH3', 'LRW', 'SRW', 'KA', 'MFI', 'SR', 'NMF', 'DW', 'N2V',
       'GCN', 'GAT', 'SAGE', 'VGNAE']

RUN = 100
eta = 11

for dataset in datasets:
    workbook = Workbook()
    network = str(dataset[0])
    if network.isdigit():
        network = 'Benchmark_{}'.format(network)
    print(network)
    for lp in lps:
        dis_matrix_avg = np.zeros((len(metrics), RUN), dtype=float)
        dis_count_matrix = np.zeros((len(metrics), eta, eta), dtype=float)
        print(network, lp)
        data = dict()
        with open('../result/{0}/{1}/details.txt'.format(network, lp), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split(' ')
                r, q = int(line[0]), round(float(line[1]) * (eta - 1))
                temp = [float(x) for x in line[2:]]
                data[(r, q)] = temp
        for r in range(RUN):
            for q1 in np.arange(0, eta - 1):
                for q2 in np.arange(q1, eta):
                    dis_count_now = np.array([m1 >= m2 for m1, m2 in zip(data[(r, q1)], data[(r, q2)])])
                    dis_count_matrix[:, q1, q2] += dis_count_now
                    if q1 != q2:
                        dis_count_matrix[:, q2, q1] += dis_count_now

        discriminability = np.zeros((len(metrics), 10), dtype=float)
        for i in range(8):
            for j in range(0, 10):
                discriminability[i][j] = np.sum(dis_count_matrix[i] < (j + 1)) / (
                        eta * eta)

        sheet = workbook.create_sheet(title=lp)
        for row in discriminability.T:
            sheet.append(list(row))

    # 删除默认的Sheet
    default_sheet = workbook['Sheet']
    workbook.remove(default_sheet)
    workbook.save(filename='single/{}_discriminability.xlsx'.format(network))
