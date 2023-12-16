import torch

# Rank
class Rank():

    def __init__(self,tggat_result,top_key):
        self.tggat_result = tggat_result
        self.top_key = top_key
    #    rank_nodes
    def rank_and_filter_nodes(self):
        node_scores = torch.tensor(self.tggat_result)
        max_scores = torch.max(node_scores, dim=1).values

        #Normalisasi skor jika perlu
        max_scores = max_scores / max_scores.sum()

        _, indices = max_scores.sort(descending=True)
        top_indices = indices[:self.top_key].tolist()
        # Ambil max_score untuk setiap indeks dalam top_indices
        max_scores_for_top_indices = max_scores[top_indices]

        return top_indices, max_scores_for_top_indices

    def get_all_nodes_scores(self):

        node_scores = torch.tensor(self.tggat_result)

        max_scores = torch.max(node_scores, dim=1).values

        # Normalisasi skor jika perlu
        max_scores = max_scores / max_scores.sum()

        # Mendapatkan skor untuk semua indeks
        all_scores = max_scores.tolist()
       
        return all_scores

    def map_indices_to_chunks(self, salient_indices, noun_phrase_index):
   
        mapped_chunks = [noun_phrase_index.get(index, '') for index in salient_indices]
        
        return mapped_chunks

    def rank_nodes_list(self, tggat_results):
        salient_nodes_indices = []
        for result in tggat_results:
            node_scores = torch.tensor(result)
            max_scores = torch.max(node_scores, dim=1).values
            #Normalisasi skor jika perlu
            max_scores = max_scores / max_scores.sum()
            _, indices = max_scores.sort(descending=True)
            top_indices = indices[:self.top_key].tolist()
            salient_nodes_indices.append(top_indices)
        return salient_nodes_indices



