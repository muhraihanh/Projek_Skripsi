import torch
from natsort import natsorted
import re
import os
import pandas as pd
import os.path as path
from PreProcessing import PreProcessing
from Embedding import EmbeddingProcessor
from NTM import LDAModel
from AnchorAwareGraph import A2G
from TgGAT import TgGATLayer
from RankFilter import Ranking_Filtering
from Evaluation import KeyphraseEvaluator



# ======================================== 1. Mengambil Dataset Abstrak
folder_path_abstract = 'Dataset\eks'
file_names = natsorted([file for file in os.listdir(folder_path_abstract) if file.lower().endswith(".txt")][:5])
judul_list = []
abstrak_list = []

for file_name in file_names:
    file_path = os.path.join(folder_path_abstract, file_name)
    with open(file_path, 'r') as file:
        content = file.read()
        # Hilangkan ".txt" dari judul
        judul = file_name.replace(".txt", "")
        # Hilangkan angka di depan judul
        judul = re.sub(r'^\d+\.', '', judul)
        judul_list.append(judul)
        abstrak = content
        abstrak_list.append(abstrak)
dataset_indo = pd.DataFrame({'Nomor': list(range(1, len(file_names) + 1)), 'Judul': judul_list, 'Abstrak': abstrak_list})

#  =========================================== 2. Mengambil Datset Keyword
folder_path_keyword = "Dataset\keyword"
file_names = natsorted([file for file in os.listdir(folder_path_keyword) if file.lower().endswith(".txt")][:5])
keyword_list = []

for file_name in file_names:
    file_path = os.path.join(folder_path_keyword  , file_name)
    with open(file_path, 'r') as file:
        content = file.read()
        keyword = content
        keyword_list.append(keyword)

dataset_indo = dataset_indo.assign(Keyword=keyword_list)


# ========================================== 3. PreProcessing
# Inisialisasi list untuk menyimpan hasil noun phrase chunking
indexing_function = lambda x: {i: phrase for i, phrase in enumerate(x)}
noun_phrase_list = []
judul_abstrak_list = []
pos_tag_list = []

# Iterasi melalui judul_list dan abstrak_list
for judul, abstrak in zip(judul_list, abstrak_list):
    # Membuat instance dari kelas PreProcessing
    processor = PreProcessing(judul=judul, abstrak=abstrak)
    # Memanggil metode-metode untuk melakukan preprocessing
    processor.gabung_string()
    processor.cleansing()
    processor.pos_tagging()
    processor.extract_candidates()
    judul_abstrak_list.append(processor.gabung_string)
    pos_tag_list.append(processor.pos_tagging)
    # Menyimpan hasil preprocessing
    # Filter hanya token yang bukan karakter titik
    filtered_noun_phrases = [phrase for phrase in processor.noun_phrase if phrase != '.']
    # Menghapus kata-kata yang sama dalam satu daftar noun phrase
    unique_noun_phrases = list(set(filtered_noun_phrases))
    noun_phrase_list.append(unique_noun_phrases)
    
dataset_indo['Judul_Gabung_Abstrak'] = judul_abstrak_list
dataset_indo['Pos_Tag'] = pos_tag_list
dataset_indo['Noun_Phrase'] = noun_phrase_list
# Membuat versi dengan indeks dari NP chunking yang telah dibersihkan
dataset_indo['np_chunking_with_index'] = dataset_indo['Noun_Phrase'].apply(lambda x: {i: phrase for i, phrase in enumerate(x)})
# print(dataset_indo)


# ================================================= 4. Embedding dengan RoBERTa
hasil_embedding_list = []
# Iterasi melalui setiap dokumen
for noun_phrase in noun_phrase_list:
    embedding_processor = EmbeddingProcessor(noun_phrase=noun_phrase)
    embedding_processor.process_embedding()
    hasil_embedding_list.append(embedding_processor.hasil_embedding)

# # Menampilkan hasil embedding
# for idx, hasil_embedding in enumerate(hasil_embedding_list):
#     av_shape = hasil_embedding.size()
#     print(f"Judul {idx + 1}- shape : {av_shape} - Hasil Embedding: {hasil_embedding.shape}")



# =========================================== 5. Neural Topic Module dengan LDA
ntm = LDAModel(num_topics=10, passes=15, no_below=2, no_above=0.5, keep_n=100000)
topic_distributions_list = []
topic_representation_index_list = []
for noun_phrase in noun_phrase_list:
    corpus = ntm.build_lda_model(noun_phrase)
    topic_distribution, topic_representation_index = ntm.get_topic_distributions(corpus)
    topic_distributions_list.append(topic_distribution)
    topic_representation_index_list.append(topic_representation_index)


# =================================================== 6. Anchor Aware Graph
anchor_graph_list = []
adj_matrix_list = []
graph = A2G(noun_phrase_list, hasil_embedding_list)
for i in range(len(noun_phrase_list)):
    anchor_aware_graph, adj_matrix = graph.build_anchor_graph_list(i)
    anchor_graph_list.append(anchor_aware_graph)
    adj_matrix_list.append(adj_matrix)


# ======================================================= 7. TgGAT
input_list = []
i = 0
for _ in anchor_graph_list:
  input_list.append(len(anchor_graph_list[i]))
  i += 1

in_features_list = input_list  # Replace ... with the values for other documents
out_features_list = 10 # Replace ... with the values for other documents, jumlah representeasi topik
dropout_list = 0.6 # Replace ... with the values for other documents
alpha_list = 0.2  # Replace ... with the values for other documents
num_heads_list = 8 # Replace ... with the values for other documents
topic_representation_dim = 10

output_list = []

# Assuming topic_distributions_list is a list of NumPy arrays
for doc_idx in range(len(anchor_graph_list)):
    # Assuming input_data is a NumPy array
    input_data = torch.tensor(anchor_graph_list[doc_idx], dtype=torch.float32)

    # Assuming adjacency_matrix is a NumPy array
    adjacency_matrix = torch.tensor(adj_matrix_list[doc_idx], dtype=torch.float32)

    # Assuming topic_representation is a NumPy array
    topic_representation = torch.tensor(topic_distributions_list[doc_idx], dtype=torch.float32)

    # Create TgGAT layer with document-specific parameters
    gat_layer = TgGATLayer(
        in_features_list[doc_idx], out_features_list,
        dropout_list, alpha_list, num_heads_list, topic_representation_dim
    )

    output = gat_layer(input_data, adjacency_matrix, topic_representation)
    output_list.append(output)

# print(output_list)


# ====================================================== 8. Rank dan Filter
# Fungsi untuk melakukan ranking dan filtering dengan metrik berbeda
def rank_and_filter_nodes(tggat_results, top_n=5):
    salient_nodes_indices = []
    for result in tggat_results:
        node_scores = torch.tensor(result)
        max_scores = torch.max(node_scores, dim=1).values
        #Normalisasi skor jika perlu
        max_scores = max_scores / max_scores.sum()
        _, indices = max_scores.sort(descending=True)
        top_indices = indices[:top_n].tolist()
        salient_nodes_indices.append(top_indices)
    return salient_nodes_indices

top_n = 5
# Menerapkan fungsi ranking dan filtering dengan metrik baru
salient_nodes_index_list = rank_and_filter_nodes(output_list, top_n=top_n)
dataset_indo['salient_nodes_index'] = salient_nodes_index_list


# ===================================================== 9. Melihat semua F1 Score setiap dokumen
def get_all_nodes_scores(tggat_results):
    all_nodes_scores = []

    for result in tggat_results:
        node_scores = torch.tensor(result)

        max_scores = torch.max(node_scores, dim=1).values

        # Normalisasi skor jika perlu
        max_scores = max_scores / max_scores.sum()

        # Mendapatkan skor untuk semua indeks
        all_scores = max_scores.tolist()
        all_nodes_scores.append(all_scores)

    return all_nodes_scores

# Contoh penggunaan
all_nodes_scores_list = get_all_nodes_scores(output_list)
dataset_indo['all_nodes_scores'] = all_nodes_scores_list

# Mencetak skor untuk setiap indeks di setiap dokumen
# for doc_idx, scores in enumerate(dataset_indo['all_nodes_scores']):
#     print(f"Dokumen {doc_idx+1}:")
#     for idx, score in enumerate(scores):
#         print(f"Indeks: {idx}, Skor: {score:.5f}")


# ======================================================== 10. Mapping
def map_indices_to_chunks_in_dataframe(df):
    
    mappings = []
    for index, row in df.iterrows():
        salient_indices = row['salient_nodes_index']
        chunk_mapping = row['np_chunking_with_index']
        mapped_chunks = [chunk_mapping.get(index, '') for index in salient_indices]
        mappings.append(mapped_chunks)
    df['mapping'] = mappings
    return df

# Apply the mapping function to the dataframe
map_indices_to_chunks_in_dataframe(dataset_indo)
dataset_indo['mapping']

# ======================================================== 11. F1 Score
n = len(dataset_indo)
# Inisialisasi kolom baru dalam dataframe
dataset_indo['precision'] = 0
dataset_indo['recall'] = 0
dataset_indo['f1score'] = 0

for i in range(n):
    extracted_keyphrases = dataset_indo['mapping'].iloc[i]
    golden_keyphrases = dataset_indo['Keyword'].iloc[i]
    golden_keyphrases_list = golden_keyphrases.lower().split(';')

    true_positives = sum(phrase.lower() in golden_keyphrases_list for phrase in extracted_keyphrases)
    false_positives = len(extracted_keyphrases) - true_positives
    false_negatives = len(golden_keyphrases_list) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Simpan hasil ke dalam dataframe
    dataset_indo.at[i, 'precision'] = precision
    dataset_indo.at[i, 'recall'] = recall
    dataset_indo.at[i, 'f1score'] = f1_score

maxf1score = dataset_indo['f1score'].max()
# print(maxf1score)

# Menghitung rata-rata precision, recall, dan f1-score
avg_precision = dataset_indo['precision'].mean()
avg_recall = dataset_indo['recall'].mean()
avg_f1score = dataset_indo['f1score'].mean()

# Menampilkan hasil rata-rata
print("Rata-rata Precision:", avg_precision)
print("Rata-rata Recall:", avg_recall)
print("Rata-rata F1-Score:", avg_f1score)














# judul = '1.Analisis Forensik Layanan Signal Private Messenger pada Smartwatch Menggunakan Metode National Institute of Justice'
# abstrak = 'Teknologi informasi dan komunikasi yang pesat berdampak positif yaitu berupa kecepatan dan kemudahan untuk mendapatkan informasi dan data dengan penggunaan smartphone yang berbasis android. Tetapi, disisi lain juga memberikan dampak negatif berupa kejahatan siber. Penelitian ini bertujuan untuk mengangkat bukti data dan informasi digital dalam mengidentifikasi kasus kejahatan pornografi dengan memakai aplikasi Signal Private Messenger (SPM) pada smartwatch selaku perangkat korespondensii. Proses analisis forensics pada pengkajian ini memakai metode National Insitute of Justice (NIJ) dan tiga tools forensik, yaitu Wondershare Dr. Fone, MOBILedit, dan Oxygen Forensics. Skenario penelitian adalah terjadinya tindak pidana pornografi. Penelitian ini berhasil mengangkat barang bukti berupa photos, videos, music, contact, dan chatt. Pengujian dengan Wondershare Dr. Fone dihasilkan akurasi 80%, MobilEdit dihasilkan akurasi 40%, dan Oxygen dihasilkan akurasi 0%. Oleh karena itu disimpulkan bahwa Wondershare Dr. Fone mmemberikan hasil yang terbaik.'
# golden_key = 'Forensics; Tools; SPM; NIJ; Smartwatch'


# # function for indexing
# indexing_function = lambda x: {i: phrase for i, phrase in enumerate(x)}

# # 1. PreProcessing
# process = PreProcessing(judul,abstrak)
# process.gabung_string()
# process.cleansing()
# process.pos_tagging()
# process.extract_candidates()
# # menghilangkan titik yang menjadi tokens
# filtered_noun_phrases = [phrase for phrase in process.noun_phrase if phrase != '.']
# # Menghapus kata-kata yang sama dalam satu daftar noun phrase
# unique_noun_phrases = list(set(filtered_noun_phrases))
# noun_phrase = unique_noun_phrases
# noun_phrase_index = indexing_function(noun_phrase)
# print(noun_phrase_index)


# # 2. Embedding with RoBERTa
# embedding = EmbeddingProcessor(noun_phrase)
# embedding.process_embedding()
# hasil_embedding = embedding.hasil_embedding
# hasil_embedding_index = indexing_function(hasil_embedding)
# print(f'Hasil Embedding ~ {hasil_embedding.shape}')


# # 3. Neural Topic Module
# ntm = LDAModel(num_topics=10, passes=15, no_below=2, no_above=0.5, keep_n=100000)
# corpus = ntm.build_lda_model(noun_phrase)
# topic_distribution, topic_representation_index = ntm.get_topic_distributions(corpus)
# print(f'Hasil Topik Distribution ~ {topic_representation_index}')


# # 4. Anchor Aware Graph
# graph = A2G(noun_phrase, hasil_embedding)
# anchor_aware_graph, adj_matrix = graph.build_anchor_graph()
# print(f'Anchor Aware Graph ~ {anchor_aware_graph.shape}')


# # 5. Topic Guided Graph Attention Network
# in_features = len(anchor_aware_graph)  # Replace ... with the values for other documents
# out_features = 10 # Replace ... with the values for other documents, jumlah representeasi topik
# dropout = 0.6 # Replace ... with the values for other documents
# alpha = 0.2  # Replace ... with the values for other documents
# num_heads = 8 # Replace ... with the values for other documents
# topic_representation_dim = 10

# input_data = torch.tensor(anchor_aware_graph)
# adjacency_matrix = torch.tensor(adj_matrix)
# topic_representation = torch.tensor(topic_distribution)

# tggat = TgGATLayer(in_features, out_features,dropout, alpha, num_heads, topic_representation_dim)
# output = tggat.forward(input_data, adjacency_matrix, topic_representation)


# # 6. Rank and Filter
# top_k = 5
# rank_filter = Ranking_Filtering(output, top_k)
# top_indices = rank_filter.rank_and_filter_nodes()
# all_scores = rank_filter.get_all_nodes_scores()
# print(top_indices)
# for idx, score in enumerate(all_scores):
#         print(f"Indeks: {idx}, Skor: {score:.5f}")

 
# # 7 Mapping
# mapped_chunks_result = rank_filter.map_indices_to_chunks(salient_indices = top_indices, noun_phrase_index = noun_phrase_index)
# print(mapped_chunks_result)


# # 8. Evaluation (Recall, Precision, dan F1-Score)
# evaluate = KeyphraseEvaluator(mapped_chunks_result, golden_key)
# evaluate.evaluate_keyphrases()

# results = evaluate.get_results()
# print(results)








# # Example usage:
# input_list = []
# i = 0
# for _ in anchor_graph_list:
#     input_list.append(len(anchor_graph_list[i]))
#     i += 1
# # anchor_graph_list, adj_matrix_list, topic_distributions_list are assumed to be lists of NumPy arrays
# # Each element of the list corresponds to the data for one document
# # out_features_list dan topic_representation_dim nilai nya sama
# in_features_list = input_list  # Replace ... with the values for other documents
# # Replace ... with the values for other documents, jumlah representeasi topik
# out_features_list = 10
# dropout_list = 0.4  # Replace ... with the values for other documents
# alpha_list = 0.1  # Replace ... with the values for other documents
# num_heads_list = 8  # Replace ... with the values for other documents
# topic_representation_dim = 10

# output_list = []

# # Assuming topic_distributions_list is a list of NumPy arrays
# for doc_idx in range(len(anchor_graph_list)):
#     # Assuming input_data is a NumPy array
#     input_data = torch.tensor(anchor_graph_list[doc_idx], dtype=torch.float32)

#     # Assuming adjacency_matrix is a NumPy array
#     adjacency_matrix = torch.tensor(
#         adj_matrix_list[doc_idx], dtype=torch.float32)

#     # Assuming topic_representation is a NumPy array
#     topic_representation = torch.tensor(
#         dataset_indo['ntm'][doc_idx], dtype=torch.float32)

#     # Create TgGAT layer with document-specific parameters
#     gat_layer = TgGATLayer(
#         in_features_list[doc_idx], out_features_list,
#         dropout_list, alpha_list, num_heads_list, topic_representation_dim
#     )

#     output = gat_layer(input_data, adjacency_matrix, topic_representation)
#     output_list.append(output)

#     # print(f"Output shape for Document {doc_idx + 1}: {output.shape}")

# # output_list now contains the GAT layer outputs for each document


# # Assuming output_list is a list of PyTorch tensors
# output_list_np = [output.detach().numpy() for output in output_list]

# # Adding the list of NumPy arrays to the dataset
# dataset_indo['tggat'] = output_list_np
