# from torch import nn
# import torch
# from sentence_transformers import SentenceTransformer, util
# from annoy import AnnoyIndex
# import numpy as np
# import pandas as pd
# import json
# import time
# from scipy.special import softmax
#
# from utils import json_from_pandas_to_main_format
#
#
# class TextModel:
#
#     def __init__(self):
#         self.model = SentenceTransformer("multi-qa-distilbert-dot-v1")
#         self.f = 768
#         self.train_data = pd.read_csv('all_cities.csv')
#         self.u1 = AnnoyIndex(self.f, 'dot')
#         self.u1.load('annoy_names.ann')  # super fast, will just mmap the file
#         self.u2 = AnnoyIndex(self.f, 'dot')
#         self.u2.load('annoy_full_city_kind.ann')  # super fast, will just mmap the file
#
#     def get_prediction_embedding(self, query, city=None):
#         query_embedding = self.model.encode(query)
#         if city:
#             query_embedding += self.model.encode(city)
#         return query_embedding
#
#     # annoy
#     def find_annoy_k_names(self, query_embedding, k=10):
#         return self.u1.get_nns_by_vector(query_embedding, k, include_distances=True)  # include_distances = True
#
#     def find_annoy_k_by_kinds(self, query_embading, k=10):
#         return self.u2.get_nns_by_vector(query_embading, k, include_distances=True)  # include_distances = True
#
#     def get_final_indexes(self, query, city=None):
#         embaded_query = self.get_prediction_embedding(query, city)
#         best_names_indexes = np.array(self.find_annoy_k_names(embaded_query, k=8)).T
#         best_kinds_indexes = np.array(self.find_annoy_k_by_kinds(embaded_query, k=8)).T
#         all_best_indexes = np.vstack([best_names_indexes, best_kinds_indexes])
#         all_best_indexes_dataframe = pd.DataFrame(all_best_indexes, columns=['ind', 'score'])
#         all_best_indexes_dataframe['ind'] = all_best_indexes_dataframe.ind.astype(np.int64)
#         all_best_unique_indexes = all_best_indexes_dataframe.drop_duplicates(['ind'])
#         all_best_unique_indexes_sorted = all_best_unique_indexes.sort_values(by=['score']).iloc[::-1]
#         all_best_unique_indexes_sorted['score'] = softmax(all_best_unique_indexes_sorted['score'])
#         return all_best_unique_indexes_sorted  # np.array(all_best_unique_indexes_sorted['ind']).transform(softmax)
#
#     def get_final_objects(self, indexes, data_table):
#         return data_table.loc[indexes]
#
#     def predict(self, query, city):
#         select_city = {0: None, 1: 'Нижний Новгород', 2: 'Ярославль', 3: 'Екатеринбург', 4: 'Владимир'}
#         city = select_city[city]
#         final_indexes = self.get_final_indexes(query, city)
#         final_objects = pd.concat(
#             [self.get_final_objects(final_indexes['ind'], self.train_data).reset_index(drop=True), final_indexes['score']], axis=1)
#         return json_from_pandas_to_main_format(final_objects.to_json(orient='records', force_ascii=False))
#
#
