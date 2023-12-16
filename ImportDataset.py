
import re
import os



class Importdata:
    def __init__(self, folder_path_abstract, folder_path_keyword):
        self.folder_path_abstract = folder_path_abstract
        self.folder_path_keyword = folder_path_keyword

    def getAbstrak(self, file_names):
        judul_list = []
        abstrak_list = []

        for file_name in file_names:
            file_path = os.path.join(self.folder_path_abstract, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                # Hilangkan ".txt" dari judul
                judul = file_name.replace(".txt", "")
                # Hilangkan angka di depan judul
                judul = re.sub(r'^\d+\.', '', judul)
                judul_list.append(judul)
                abstrak = content
                abstrak_list.append(abstrak)

        return judul_list, abstrak_list

    def getKeyword(self, file_names_keyword):
        keyword_list = []

        for file_name in file_names_keyword:
            file_path = os.path.join(self.folder_path_keyword, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                keyword = content
                keyword_list.append(keyword)

        return keyword_list


# ======================================== 1. Mengambil Dataset Abstrak
# def getAbstrak(folder_path_abstract_input, file_names_input):
#     folder_path_abstract = folder_path_abstract_input
#     file_names = file_names_input
#     judul_list = []
#     abstrak_list = []

#     for file_name in file_names:
#         file_path = os.path.join(folder_path_abstract, file_name)
#         with open(file_path, 'r') as file:
#             content = file.read()
#             # Hilangkan ".txt" dari judul
#             judul = file_name.replace(".txt", "")
#             # Hilangkan angka di depan judul
#             judul = re.sub(r'^\d+\.', '', judul)
#             judul_list.append(judul)
#             abstrak = content
#             abstrak_list.append(abstrak)
    
#     return judul_list, abstrak_list

# def getKeyword(folder_path_keyword_input, file_names_keyword_input):
#     #  =========================================== 2. Mengambil Datset Keyword
#     folder_path_keyword = folder_path_keyword_input
#     file_name_keyword = file_names_keyword_input
#     keyword_list = []

#     for file_name in file_name_keyword:
#         file_path = os.path.join(folder_path_keyword  , file_name)
#         with open(file_path, 'r') as file:
#             content = file.read()
#             keyword = content
#             keyword_list.append(keyword)
    
#     return keyword_list


# 