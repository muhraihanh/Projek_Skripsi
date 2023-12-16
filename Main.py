import torch
from natsort import natsorted
import os
import pandas as pd
import os.path 
import streamlit as st

from GUI import Gui
from ImportDataset import Importdata
from PraPemrosesan import PraPemrosesan
from Embedding import Embedding
from NTM import LDAModel
from AnchorAwareGraph import A2G
from TgGAT import TgGATLayer
from RankFilter import Rank
# from Evaluation import Evaluation


st.set_page_config(page_title="Keyphrase Extraction", page_icon="📚", layout="wide")

# ---- HEADER SECTION ----
with st.container():
    st.title("Keyphrase Extraction")
    st.write("✨Keyphrase Extraction menggunakan RoBERTa dan Topic Guided Graph Attention Network✨")


with st.container():
    st.write("---")
 
   

# Sidebar with links
page = st.sidebar.selectbox("Kategori", ["Demonstrasi", "Evaluasi"])

# Main content based on the selected page
if page == "Demonstrasi":
    st.subheader("➤ Demonstrasi")
    # init kelas GUI
    main = Gui()
    input = main.input()

    # function for indexing
    indexing_function = lambda x: {i: phrase for i, phrase in enumerate(x)}


    
    # Data Uji 
    # Judul = 60.Analisis Perbandingan Tiga Metode Untuk Mendiagnosa Penyakit Mata Pada Manusia.txt
    
    # Abstrak = Untuk mendiagnosa penyakit mata pada manusia diperlukan perhitungan probabilitas
    # yang terbaik. Karena mata merupakan salah satu bagian terpenting pada tubuh manusia yang
    # harus di jaga kesehatannya. Penelitian ini bertujuan untuk menganalisis perbandingan dari 3
    # metode diantaranya : metode Case-Based Reasoning, Naïve Bayes dan Certainty Factor
    # sehingga bisa diketahui metode mana yang terbaik untuk melakukan pendiagnosaan. Setelah
    # melakukan perbandingan, untuk perhitungan metode Case-Based Reasoning didapatkan hasil
    # probabilitas 61,6 %, metode Naïve Bayes didapatkan hasil 56,36% dan metode Certainty
    # Factor didapatkan hasil 90,4%. Dapat disimpulkan, metode Certainty Factor adalah metode
    # yang terbaik untuk melakukan pendiagnosaan penyakit mata pada manusia. Setelah itu, akan
    # dibuatkan suatu sistem pakar menggunakan metode Certainty Factor untuk mendiagnosa
    # penyakit mata pada manusia. Sistem pakar merupakan peniru suatu pakar dalam melakukan
    # diagnosis suatu penyakit. Tujuan dibuatkan sistem pakar ini, supaya dapat membantu pasien
    # untuk mendiagnosa jenis penyakit mata apa berdasarkan gejala gejala yang dialaminya..
    
    # Keyword = case-based reasoning;certainty factor;diagnose;naïve bayes;penyakit mata;sistem pakar

    


    if st.button("Extract ❗"):
        if main.judul and main.abstrak:
        # 1. PreProcessing
            process = PraPemrosesan(main.judul, main.abstrak)
            process.gabung_string()
            process.cleansing()
            process.pos_tagging()
            # candidates()
            process.extract() # menghasilkan process.noun_phrase
            #  menghilangkan titik yang menjadi tokens
            filtered_noun_phrases = [phrase for phrase in process.noun_phrase if phrase != '.']
            # Menghapus kata-kata yang sama dalam satu daftar noun phrase
            unique_noun_phrases = list(set(filtered_noun_phrases))
            noun_phrase = unique_noun_phrases
            noun_phrase_index = indexing_function(noun_phrase)
            
            # st.write("POS Tagged Tokens:", process.pos_tag)
            # st.write("Noun Phrases :")
            # st.write(noun_phrase)
            

            # 2. Embedding with RoBERTa
            embedding = Embedding(noun_phrase)
            embedding.process_embedding()
            hasil_embedding = embedding.hasil_embedding
            hasil_embedding_index = indexing_function(hasil_embedding)
            # st.write("Hasil Embedding ~ Shape : ", hasil_embedding.shape)
            # st.write(hasil_embedding)


            # 3. Neural Topic Module
            ntm = LDAModel(num_topics=10, passes=15, no_below=2, no_above=0.5, keep_n=100000)
            corpus = ntm.build_lda_model(noun_phrase)
            topic_distribution, topic_representation_index = ntm.get_topic_distributions(corpus)
            baris = len(topic_distribution)
            kolom = len(topic_distribution[0])
            # st.write(f"Hasil Neural Topic Module ~ Shape : ({baris}, {kolom})")
            # st.write(topic_distribution)


            # 4. Anchor Aware Graph
            graph = A2G(noun_phrase, hasil_embedding)
            anchor_aware_graph, adj_matrix = graph.build_anchor_graph()
            # st.write("Hasil Anchor Aware Graph ~ Shape : ", anchor_aware_graph.shape)
            # st.write(anchor_aware_graph)


            # 5. Topic Guided Graph Attention Network
            in_features = len(anchor_aware_graph)  # Replace ... with the values for other documents
            out_features = 10 # Replace ... with the values for other documents, jumlah representeasi topik
            dropout = 0.6 # Replace ... with the values for other documents
            alpha = 0.2  # Replace ... with the values for other documents
            num_heads = 8 # Replace ... with the values for other documents
            topic_representation_dim = 10

            input_data = torch.tensor(anchor_aware_graph)
            adjacency_matrix = torch.tensor(adj_matrix)
            topic_representation = torch.tensor(topic_distribution)

            tggat = TgGATLayer(in_features, out_features,dropout, alpha, num_heads, topic_representation_dim)
            # forward = tggat_process(), output = node_score
            output = tggat.forward(input_data, adjacency_matrix, topic_representation)
            # st.write("Hasil TgGAT ~ Shape : ", output.shape)
            # st.write(output)

            # 6. Rank and Filter
            top_k = main.jumlah_key
            rank_filter = Rank(output, top_k)
            top_indices, score_indices = rank_filter.rank_and_filter_nodes()
            all_scores = rank_filter.get_all_nodes_scores()
            # print(top_indices)
            # for idx, score in enumerate(all_scores):
            #         print(f"{noun_phrase[idx]}, Skor: {score}")

            
            # 7 Mapping
            mapped_chunks_result = rank_filter.map_indices_to_chunks(salient_indices = top_indices, noun_phrase_index = noun_phrase_index)
            
            
            # # 8. Evaluation (Recall, Precision, dan F1-Score)
            # evaluate = KeyphraseEvaluator(mapped_chunks_result, main.golden_keyword)
            # evaluate.evaluate_keyphrases()

            with st.container():
                st.write("---")

            # menampilkan keyphrase sistem dan score nya
            st.subheader("☑️Extracted Keyphrase From System")
            for i,item in enumerate(mapped_chunks_result):
                  # Ambil nilai pada indeks 1
                st.write(f"{i+1}. {item}   ~ Score: {score_indices[i]}  ")
            
            

            # st.subheader("Hasil Evaluasi Model")
            # st.write("Precision: ", evaluate.precision)
            # st.write("Recall:", evaluate.recall)
            # st.write("F1-Score:", evaluate.f1_score)

            # st.write(' CONFUSION MATRIX  ')
            # st.write("True Positives (TP):", TP)
            # st.write("True Negatives (TN):", TN)
            # st.write("False Positives (FP):", FP)
            # st.write("False Negatives (FN):", FN)
            # st.write("Precision: ", self.precision)
            # st.write("Recall:", self.recall)
            # # st.write("Accuracy:", self.accuracy)
            # st.write("F1-Score:", self.f1_score)
    else:
        st.warning("Masukkan Judul, Abstrak, Jumlah Key terlebih dahulu.")


elif page == "Evaluasi":
   
    st.subheader("➤ Evaluasi")
    folder_path_abstract = 'Dataset\eks'
    folder_path_keyword = "Dataset\keyword"
    data = Importdata(folder_path_abstract, folder_path_keyword)
    
    file_name_abstract = natsorted([file for file in os.listdir(folder_path_abstract) if file.lower().endswith(".txt")])
    
    judul_list, abstrak_list = data.getAbstrak(file_name_abstract)
    dataset_indo = pd.DataFrame({'Nomor': list(range(1, len(file_name_abstract) + 1)), 'Judul': judul_list, 'Abstrak': abstrak_list})

    file_name_keyword = natsorted([file for file in os.listdir(folder_path_keyword) if file.lower().endswith(".txt")])
    keyword_list = data.getKeyword(file_name_keyword)
    dataset_indo = dataset_indo.assign(Keyword=keyword_list)
    
    st.write(dataset_indo)
   


    if st.button("Evaluate Model"): #==================================================================
        # ========================================== 3. PreProcessing
        # Inisialisasi list untuk menyimpan hasil noun phrase chunking
        indexing_function = lambda x: {i: phrase for i, phrase in enumerate(x)}
        noun_phrase_list = []
        judul_abstrak_list = []
        pos_tag_list = []

        # Iterasi melalui judul_list dan abstrak_list
        for judul, abstrak in zip(judul_list, abstrak_list):
            # Membuat instance dari kelas PreProcessing
            processor = PraPemrosesan(judul=judul, abstrak=abstrak)
            # Memanggil metode-metode untuk melakukan preprocessing
            processor.gabung_string()
            processor.cleansing()
            processor.pos_tagging()
            processor.extract()
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
            embedding_processor = Embedding(noun_phrase=noun_phrase)
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
        top_n = 7
        rank = Rank(output_list, top_n)
        
        # Menerapkan fungsi ranking dan filtering dengan metrik baru
        salient_nodes_index_list = rank.rank_nodes_list(output_list)
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
            df['choosen_keyphrase'] = mappings
            return df

        # Apply the mapping function to the dataframe
        # return dataset_indo['mapping']
        map_indices_to_chunks_in_dataframe(dataset_indo)
       

        # ======================================================== 11. F1 Score
        n = len(dataset_indo)
        # Inisialisasi kolom baru dalam dataframe
      
        dataset_indo['precision'] = 0
        dataset_indo['recall'] = 0
        dataset_indo['f1score'] = 0
        dataset_indo['TP'] = 0
        dataset_indo['FP'] = 0
        dataset_indo['FN'] = 0


        for i in range(n):
            extracted_keyphrases = dataset_indo['choosen_keyphrase'].iloc[i]
            golden_keyphrases = dataset_indo['Keyword'].iloc[i]
            golden_keyphrases_list = golden_keyphrases.lower().split(';')

            true_positives = sum(phrase.lower() in golden_keyphrases_list for phrase in extracted_keyphrases)

            false_positives = len(extracted_keyphrases) - true_positives

            false_negatives = len(golden_keyphrases_list) - true_positives
            # print(false_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Simpan hasil ke dalam dataframe
            dataset_indo.at[i, 'TP'] = true_positives
            dataset_indo.at[i, 'FP'] = false_positives
            dataset_indo.at[i, 'FN'] = false_negatives
            dataset_indo.at[i, 'precision'] = precision
            dataset_indo.at[i, 'recall'] = recall
            dataset_indo.at[i, 'f1score'] = f1_score

        maxf1score = dataset_indo['f1score'].max()
        st.write("Nilai F1-Score terbesar",maxf1score)

        # Menghitung rata-rata precision, recall, dan f1-score
        avg_TP = dataset_indo['TP'].mean()
        avg_FP = dataset_indo['FP'].mean()
        avg_FN = dataset_indo['FN'].mean()
        avg_precision = dataset_indo['precision'].mean()
        avg_recall = dataset_indo['recall'].mean()
        avg_f1score = dataset_indo['f1score'].mean()

        # Menampilkan hasil rata-rata
        st.write("☑️Confusion Matrix")
        st.write("Rata-rata TP:", avg_TP)
        st.write("Rata-rata FP:", avg_FP)
        st.write("Rata-rata FN:", avg_FN)
        
        st.write("☑️Metrik Performa")
        st.write("Rata-rata Precision:", avg_precision)
        st.write("Rata-rata Recall:", avg_recall)
        st.write("Rata-rata F1-Score:", avg_f1score)



