import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import negative_sampling

from torch_geometric.nn import GAE, VGAE

from modules.MyGCN import MyGCN
from modules.MyGAT import MyGAT
from modules.MySAGE import MySAGE
from modules.MyVGNAE import MyVGNAE


# --------------------------------------------
def Common_Neighbors(matrix, pred_index):
    sim_matrix = np.dot(matrix, matrix)
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


# --------------------------------------------
def Resource_Allocation(matrix, pred_index):
    add_row = np.sum(matrix, axis=1)
    add_row = add_row[:, np.newaxis]
    sim_matrix = matrix / add_row
    sim_matrix = np.nan_to_num(sim_matrix)
    sim_matrix = np.dot(matrix, sim_matrix)
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


# -------------------------------------------
def Jaccard(matrix, pred_index):
    CN = np.dot(matrix, matrix)
    union = CN * (1 - np.eye(matrix.shape[0]))
    sim_matrix = np.divide(CN, union, out=np.zeros_like(matrix), where=(union != 0))
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


# --------------------------------------------
def Katz(matrix, pred_index):
    parameter = 0.01
    eye = np.eye(matrix.shape[0])
    temp = eye - matrix * parameter
    sim_matrix = np.linalg.inv(temp)
    sim_matrix = sim_matrix - eye
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def Preferential_Attachment(matrix, pred_index):
    add_row = sum(matrix)
    sim_matrix = np.outer(add_row, add_row)
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def Cannistraci_Hebb_L2_L3(matrix, pred_index):
    def geo_mean(arr):
        row_prod = np.prod(arr, axis=1)
        return np.power(row_prod, 1.0 / arr.shape[1])

    m = len(matrix)
    ch2_matrix = np.zeros([m, m])
    ch2_l3_matrix = np.zeros([m, m])
    cn_l3_matrix = np.zeros([m, m])
    ra_l3_matrix = np.zeros([m, m])
    w = []
    ne = [[] for _ in range(m)]
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            if matrix[i][j] == 0:
                w.append([i, j])
            else:
                ne[i].append(j)
                ne[j].append(i)
    degp = np.sum(matrix, axis=1)
    for [u,v] in w:
        Au = ne[u]
        Av = ne[v]
        # L2
        inter = np.array(list(set(Au) & set(Av)), dtype=int)
        if inter.size != 0:
            deg = degp[inter]
            subnetwork = matrix[np.ix_(inter, inter)]
            iDeg = np.sum(subnetwork, axis=1)
            ch2 = sum([(id + 1) / (d - id - 1) for id, d in zip(iDeg, deg)])
            ch2_matrix[u][v] = ch2
            ch2_matrix[v][u] = ch2
        # L3
        paths = []
        cn,ra = 0,0
        for x, au in enumerate(Au):
            for y, av in enumerate(Av):
                if matrix[au][av] != 0:
                    cn += 1
                    paths.append(au)
                    paths.append(av)
                    ra += 1 / np.sqrt(degp[au] * degp[av])
        if cn:
            # axy
            cn_l3_matrix[u][v] = cn
            cn_l3_matrix[v][u] = cn
            ra_l3_matrix[u][v] = ra
            ra_l3_matrix[v][u] = ra

            paths = np.array(paths)
            paths_size = [cn, 2]
            inter, idx = np.unique(paths, return_inverse=True)
            subnetwork = matrix[:, inter][inter]
            iDeg = np.sum(subnetwork, axis=1)[idx]
            deg = degp[paths] - iDeg - matrix[:, u][paths] - matrix[:, v][paths]
            iDeg = (iDeg + 1).reshape(paths_size)
            deg = (deg + 1).reshape(paths_size)
            ch2_l3 = sum(geo_mean(iDeg) / geo_mean(deg))
            ch2_l3_matrix[u][v] = ch2_l3
            ch2_l3_matrix[v][u] = ch2_l3
    return np.array([ch2_matrix[u][v] for [u, v] in pred_index]), np.array(
        [ch2_l3_matrix[u][v] for [u, v] in pred_index]), np.array(
        [cn_l3_matrix[u][v] for [u, v] in pred_index]), np.array([ra_l3_matrix[u][v] for [u, v] in pred_index])


def Local_Random_Walk(matrix, pred_index):
    step = 3
    deg = np.sum(matrix, axis=1)
    M = np.sum(deg)
    if M == 0:
        return np.zeros(pred_index.shape[0])
    deg = deg[:, np.newaxis]
    train_data_d = np.divide(matrix, deg, out=np.zeros_like(matrix), where=deg != 0)
    sim_matrix = np.eye(matrix.shape[0])
    stepi = 0
    while stepi < step:
        sim_matrix = np.dot(train_data_d.T, sim_matrix)
        stepi += 1

    sim_matrix = sim_matrix.T * deg / M
    sim_matrix = sim_matrix + sim_matrix.T
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def Superposed_Random_Walk(matrix, pred_index):
    step = 3
    deg = np.sum(matrix, axis=1)
    M = np.sum(deg)
    if M == 0:
        return np.zeros(pred_index.shape[0])
    deg = deg[:, np.newaxis]
    train_data_d = np.divide(matrix, deg, out=np.zeros_like(matrix), where=deg != 0)
    tempsim = np.eye(matrix.shape[0])
    sim_matrix = np.zeros_like(matrix)
    stepi = 0
    while stepi < step:
        tempsim = np.dot(train_data_d, tempsim)
        sim_matrix = sim_matrix + tempsim
        stepi += 1

    sim_matrix = sim_matrix.T * deg / M
    sim_matrix = sim_matrix + sim_matrix.T
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def Average_Commute_Time(matrix, pred_index):
    D = np.eye(matrix.shape[0])
    D[np.diag_indices(D.shape[0])] = np.sum(matrix, axis=1)

    pinvL = np.linalg.pinv(D - matrix)

    Lxx = np.diag(pinvL)
    Lxx = np.tile(Lxx, (matrix.shape[0], 1))
    Lxx = Lxx + Lxx.T - 2 * pinvL
    sim_matrix = np.divide(1, Lxx, out=np.zeros_like(pinvL), where=~np.isclose(Lxx, 0))
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def Matrix_Forest_Index(matrix, pred_index):
    I = np.eye(matrix.shape[0])
    D = I.copy()
    D[np.diag_indices(D.shape[0])] = np.sum(matrix, axis=1)
    L = D - matrix
    sim_matrix = np.linalg.inv(I + L)
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def simRank(G, pred_index):
    sim_matrix = nx.simrank_similarity(G)
    return np.array([sim_matrix[u][v] for [u, v] in pred_index])


def sim_DeepWalk(G, pred_index):
    from karateclub.node_embedding.neighbourhood import DeepWalk
    deepwalk = DeepWalk(walk_number=10,walk_length=40,dimensions=32,window_size=10,min_count=0)
    deepwalk.fit(G)
    z =deepwalk.get_embedding()
    edge_label_index = pred_index.T
    sim = np.sum(z[edge_label_index[0]] * z[edge_label_index[1]], axis=-1)
    return sim


def sim_Node2Vec(G, pred_index):
    from karateclub.node_embedding.neighbourhood import Node2Vec
    node2Vec = Node2Vec(walk_number=10,walk_length=40,dimensions=32,window_size=10,p=1,q=0.25,min_count=0)
    node2Vec.fit(G)
    z =node2Vec.get_embedding()
    edge_label_index = pred_index.T
    sim = np.sum(z[edge_label_index[0]] * z[edge_label_index[1]], axis=-1)
    return sim


def sim_NetMF(G, pred_index):
    from karateclub.node_embedding.neighbourhood import NetMF
    netmf = NetMF(dimensions=min(len(G.nodes()),128))
    netmf.fit(G)
    z=netmf.get_embedding()
    edge_label_index=pred_index.T
    sim=np.sum(z[edge_label_index[0]] * z[edge_label_index[1]], axis=-1)
    return sim


def sim_GNN(model,node_num, train_num, edge_index, test_mask, train_mask, zero_index):
    def train(x):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_index)

        neg_edge_index = negative_sampling(
            edge_index=train_index, num_nodes=node_num,
            num_neg_samples=train_num, method='sparse')

        edge_label_index = torch.cat(
            [train_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            torch.ones(train_num), torch.zeros(train_num)
        ], dim=0).to(device)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss

    def test(x, train_index, pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_index)
        pos_pred = model.decode(z, pos_edge_index).view(-1)
        neg_pred = model.decode(z, neg_edge_index).view(-1)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        return pred.cpu().numpy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = torch.from_numpy(edge_index.T).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)

    train_index=edge_index[:, train_mask]
    test_index=edge_index[:, test_mask]
    zero_index = torch.from_numpy(zero_index.T).to(device)

    x = torch.randn(node_num, 128, device=device)
    if model=='GCN':
        model = MyGCN(128,32,16).to(device)
    elif model=='GAT':
        model = MyGAT(128,32,16).to(device)
    elif model=='SAGE':
        model = MySAGE(128,32,16).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    if train_num > 0:
        for epoch in range(300):
            loss = train(x)
    sim = test(x, train_index, test_index, zero_index)
    return sim


def sim_VGNAE(model,node_num, train_num, edge_index, test_mask, train_mask, zero_index):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = torch.from_numpy(edge_index.T).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)

    train_index=edge_index[:, train_mask]
    test_index=edge_index[:, test_mask]
    zero_index = torch.from_numpy(zero_index.T).to(device)
    x = torch.randn(node_num, 128, device=device)
    if model == 'GNAE':
        model = GAE(MyVGNAE(128, 16, model)).to(device)
    if model == 'VGNAE':
        model = VGAE(MyVGNAE(128, 16, model)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x,  train_index)
        loss = model.recon_loss(z,  train_index)
        if model in ['VGNAE']:
            loss = loss + (1 / node_num) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return loss

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x,  train_index)
        pos_pred = model.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        return pred.cpu().numpy()

    if train_num > 0:
        for epoch in range(300):
            loss = train()
    sim=test(test_index, zero_index)
    return sim


def nor_sim(sim):
    max_sim = max(sim)
    min_sim = min(sim)
    if max_sim - min_sim == 0:
        return np.zeros(len(sim))
    normalized_sim = (sim - min_sim) / (max_sim - min_sim)
    return normalized_sim
