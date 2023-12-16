import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class A2G:
    def __init__(self, noun_phrase_list, hasil_embedding_list, threshold=0.6):
        self.noun_phrase_list = noun_phrase_list
        self.hasil_embedding_list = hasil_embedding_list
        self.threshold = threshold  
        self.edges_list = []
        self.matrix_adj = []

    def build_anchor_graph(self):
        # Ambil kandidat calon present keyphrase dari baris ini
        candidates = self.noun_phrase_list
        max_dim = len(candidates)

        # Ambil semua embedding dari candidates
        embeddings = np.array(self.hasil_embedding_list)

        # Hitung cosine similarity untuk semua embedding
        similarity_matrix = cosine_similarity(embeddings)

        # Tambahkan simpul (node) ke graf
        G = nx.Graph()
        G.add_nodes_from(range(len(candidates)))

        # Tambahkan edge (hubungan) berdasarkan kesamaan dengan bobot
        edges = np.column_stack(np.where(similarity_matrix > self.threshold))

        edges_with_weights = [(i, j, similarity_matrix[i][j])
                              for i, j in edges]
        G.add_weighted_edges_from(edges_with_weights)

        # Dapatkan adjacency matrix
        adj_matrix = nx.adjacency_matrix(G).todense()

        # Tambahkan padding agar dimensinya sama dengan max_dim
        pad_dim = max_dim - adj_matrix.shape[0]
        padded_matrix = np.pad(
            adj_matrix, ((0, pad_dim), (0, pad_dim)), 'constant')

        return padded_matrix, adj_matrix
    
    def build_anchor_graph_list(self, index):
        # Ambil kandidat calon present keyphrase dari baris ini
        candidates = self.noun_phrase_list[index]
        max_dim = len(candidates)
        # print(f" max_dim : {max_dim}")

        # Ambil semua embedding dari candidates
        embeddings = np.array(self.hasil_embedding_list[index])

        # Hitung cosine similarity untuk semua embedding
        similarity_matrix = cosine_similarity(embeddings)

        # Tambahkan simpul (node) ke graf
        G = nx.Graph()
        G.add_nodes_from(range(len(candidates)))

        # Tambahkan edge (hubungan) berdasarkan kesamaan dengan bobot
        edges = np.column_stack(np.where(similarity_matrix > self.threshold))

        edges_with_weights = [(i, j, similarity_matrix[i][j])
                              for i, j in edges]
        G.add_weighted_edges_from(edges_with_weights)

        # Dapatkan adjacency matrix
        adj_matrix = nx.adjacency_matrix(G).todense()

        # Tambahkan padding agar dimensinya sama dengan max_dim
        pad_dim = max_dim - adj_matrix.shape[0]
        padded_matrix = np.pad(
            adj_matrix, ((0, pad_dim), (0, pad_dim)), 'constant')

        return padded_matrix, adj_matrix