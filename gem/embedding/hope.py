disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz


class HOPE(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the HOPE class

        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        '''
        hyper_params = {
            'method_name': 'hope_gsvd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None, amatrix=None,
                        nodelist=None, weight='weight'):
        if graph is None and edge_f is None and amatrix is None:
            raise Exception('graph/edge_f/amatrix needed')

        if amatrix is None:
            if graph is None:
                graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
            if nodelist is None:
                nodelist = sorted(graph.nodes)
            A = nx.to_numpy_matrix(graph, nodelist=nodelist, weight=weight)
            n = graph.number_of_nodes()
        else:
            A = np.mat(amatrix)
            n = A.shape[0]

        if self._proximity == 'katz':
            M_g = np.eye(n) - self._beta * A
            M_l = self._beta * A
        elif self._proximity == 'common-neighbors':
            M_g = np.eye(n)
            M_l = A * A
        S = np.dot(np.linalg.inv(M_g), M_l)

        u, s, vt = lg.svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._X = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        return self._X, eig_err

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :self._d // 2], self._X[j, self._d // 2:])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


if __name__ == '__main__':
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = HOPE(4, 0.01)
    embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
    print('HOPE:\n\tTraining time: %f' % (time() - t1))

    viz.plot_embedding2D(embedding.get_embedding()[:, :2],
                         di_graph=G, node_colors=None)
    plt.show()
