# # from torch import nn
# # import torch
# # from sentence_transformers import SentenceTransformer, util
# # from annoy import AnnoyIndex
# # import numpy as np
# # import pandas as pd
# # import json
# # import time
# # from scipy.special import softmax
# #
# # from utils import json_from_pandas_to_main_format
# #
# #
# # class TextModel:
# #
# #     def __init__(self):
# #         self.model = SentenceTransformer("multi-qa-distilbert-dot-v1")
# #         self.f = 768
# #         self.train_data = pd.read_csv('all_cities.csv')
# #         self.u1 = AnnoyIndex(self.f, 'dot')
# #         self.u1.load('annoy_names.ann')  # super fast, will just mmap the file
# #         self.u2 = AnnoyIndex(self.f, 'dot')
# #         self.u2.load('annoy_full_city_kind.ann')  # super fast, will just mmap the file
# #
# #     def get_prediction_embedding(self, query, city=None):
# #         query_embedding = self.model.encode(query)
# #         if city:
# #             query_embedding += self.model.encode(city)
# #         return query_embedding
# #
# #     # annoy
# #     def find_annoy_k_names(self, query_embedding, k=10):
# #         return self.u1.get_nns_by_vector(query_embedding, k, include_distances=True)  # include_distances = True
# #
# #     def find_annoy_k_by_kinds(self, query_embading, k=10):
# #         return self.u2.get_nns_by_vector(query_embading, k, include_distances=True)  # include_distances = True
# #
# #     def get_final_indexes(self, query, city=None):
# #         embaded_query = self.get_prediction_embedding(query, city)
# #         best_names_indexes = np.array(self.find_annoy_k_names(embaded_query, k=8)).T
# #         best_kinds_indexes = np.array(self.find_annoy_k_by_kinds(embaded_query, k=8)).T
# #         all_best_indexes = np.vstack([best_names_indexes, best_kinds_indexes])
# #         all_best_indexes_dataframe = pd.DataFrame(all_best_indexes, columns=['ind', 'score'])
# #         all_best_indexes_dataframe['ind'] = all_best_indexes_dataframe.ind.astype(np.int64)
# #         all_best_unique_indexes = all_best_indexes_dataframe.drop_duplicates(['ind'])
# #         all_best_unique_indexes_sorted = all_best_unique_indexes.sort_values(by=['score']).iloc[::-1]
# #         all_best_unique_indexes_sorted['score'] = softmax(all_best_unique_indexes_sorted['score'])
# #         return all_best_unique_indexes_sorted  # np.array(all_best_unique_indexes_sorted['ind']).transform(softmax)
# #
# #     def get_final_objects(self, indexes, data_table):
# #         return data_table.loc[indexes]
# #
# #     def predict(self, query, city):
# #         select_city = {0: None, 1: 'Нижний Новгород', 2: 'Ярославль', 3: 'Екатеринбург', 4: 'Владимир'}
# #         city = select_city[city]
# #         final_indexes = self.get_final_indexes(query, city)
# #         final_objects = pd.concat(
# #             [self.get_final_objects(final_indexes['ind'], self.train_data).reset_index(drop=True), final_indexes['score']], axis=1)
# #         return json_from_pandas_to_main_format(final_objects.to_json(orient='records', force_ascii=False))
# #
# #
#
#
#
# from sentence_transformers import SentenceTransformer
# from torch import nn
# import torch
# import time
# import pickle
# import base64
# from PIL import Image
# import io
# from joblib import dump, load
# import pandas as pd
# import json
#
# from utils import json_from_pandas_to_main_format
#
#
# class TextModel:
#     def __init__(self, n_classes=371):
#         self.train_data = pd.read_csv('All_cities_super_final_version.csv')
#         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
#         #
#         for name, param in self.embedder.named_parameters():
#             param.requires_grad = False
#         #
#         self.model = nn.Sequential(
#             nn.Linear(in_features=768, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=n_classes),
#         )
#
#         self.model.load_state_dict(torch.load("models/text_model.pt", map_location=device))
#         self.model = self.model.to(device)
#         self.model.eval()
#         self.find_XID = pd.read_csv('find_XID.csv')
#         self.find_XID = self.find_XID.reindex(self.find_XID['target']).drop('target', axis=1)
#         self.inverse_transform = self.find_XID.to_dict()['XID']
#         self.forward_transform = {v: k for k, v in self.inverse_transform.items()}
#         print("model loaded")
#
#     def get_prediction_embeding(self, query, city=None):
#         query_embading = self.embedder.encode(query)
#         if city:
#             query_embading += self.embedder.encode(city)
#         return query_embading
#
#     def predict(self, text, city):
#         select_city = {0: None, 1: 'Нижний Новгород', 2: 'Ярославль', 3: 'Екатеринбург', 4: 'Владимир'}
#         city = select_city[city]
#         query = torch.tensor(self.get_prediction_embeding(text, city).reshape(1, -1))
#
#         prob = nn.Softmax(dim=1)(self.model(query)).squeeze().cpu()
#         data = self.get_topk(prob)
#         dist = self.get_dist(data)
#         return {'categories': [{'label': label, 'prob': prob} for label, prob in dist.items()],
#                 'objects': json_from_pandas_to_main_format(data.to_json(orient='records', force_ascii=False))}
#         # js = {'categories': [{'label': label, prob: prob} for label, prob in dist.items()],
#         #       'objects': data.to_json(orient='records', force_ascii=False, lines=True)}
#         # return json.dumps(js)
#
#     def get_dist(self, data):
#         dist = {'жилье': 0, 'археологические музеи': 0, 'археология': 0, 'архитектура': 0, 'художественные галереи': 0,
#                 'банк': 0, 'банки': 0, 'биографические музеи': 0, 'мосты': 0, 'места захоронений': 0, 'замки': 0,
#                 'соборы': 0, 'католические церкви': 0, 'кладбища': 0, 'детские музеи': 0, 'детские театры': 0,
#                 'церкви': 0, 'кинотеатры': 0, 'цирки': 0, 'концертные залы': 0, 'культурный': 0, 'защитные стены': 0,
#                 'разрушенные объекты': 0, 'восточные ортодоксальные церкви': 0, 'музеи моды': 0, 'продукты питания': 0,
#                 'фортификационные сооружения': 0, 'укрепленные башни': 0, 'фонтаны': 0, 'сады и парки': 0,
#                 'геологические образования': 0, 'исторические': 0, 'историческая архитектура': 0,
#                 'исторические районы': 0, 'исторические дома музеи': 0, 'исторические объекты': 0,
#                 'исторические поселения': 0, 'исторические места': 0, 'исторические музеи': 0,
#                 'промышленные объекты': 0, 'инсталляции': 0, 'интересные места': 0, 'кремли': 0, 'местные музеи': 0,
#                 'усадьбы': 0, 'военные музеи': 0, 'монастыри': 0, 'памятники': 0, 'монументы и памятники': 0,
#                 'мечети': 0, 'горные вершины': 0, 'музеи': 0, 'музеи науки и технологии': 0, 'музыкальные места': 0,
#                 'национальные музеи': 0, 'природные': 0, 'природные монументы': 0, 'природные заповедники': 0,
#                 'музеи под открытым небом': 0, 'оперные театры': 0, 'другие': 0, 'другие археологические объекты': 0,
#                 'другие мосты': 0, 'другие здания и сооружения': 0, 'другие места захоронения': 0, 'другие церкви': 0,
#                 'другие отели': 0, 'другие музеи': 0, 'другие природоохранные зоны': 0,
#                 'другие технологические музеи': 0, 'другие храмы': 0, 'другие театры': 0, 'другие башни': 0,
#                 'планетарии': 0, 'кукольные театры': 0, 'железнодорожные станции': 0, 'религия': 0, 'рестораны': 0,
#                 'скальные образования': 0, 'научные музеи': 0, 'скульптуры': 0, 'спорт': 0, 'площади': 0, 'стадионы': 0,
#                 'театры и развлечения': 0, 'туристические объекты': 0, 'туристический объект': 0, 'башни': 0,
#                 'триумфальные арки': 0, 'неклассифицированные объекты': 0, 'городская среда': 0,
#                 'смотровые площадки': 0, 'военные памятные места': 0, 'водяные башни': 0, 'зоопарки': 0, 'школы': 0,
#                 'образование': 0, 'университеты': 0, 'тюрьма': 0}
#         i = 0
#         for _, line in data.iterrows():
#             for kind in line['Kind'][1:-1].split(', '):
#                 dist[kind[1:-1]] += 1
#                 i += 1
#         if i != 0:
#             dist = {k: v / i for k, v in dist.items()}
#         return dist
#
#     def get_topk(self, pred, k=15):
#         v, i = torch.topk(pred, k)
#         vi = {i: v for i, v in zip(i.detach().numpy(), v.detach().numpy())}
#         city_names = [self.inverse_transform[ind] for ind in i.numpy()]
#         data = self.train_data.loc[self.train_data.XID.isin(city_names)]
#         scores = data['XID'].apply(lambda x: vi[self.forward_transform[x]])
#         data['score'] = scores
#         return data

from sentence_transformers import SentenceTransformer
from torch import nn
import torch
import time
import pickle
import base64
from PIL import Image
import io
from joblib import dump, load
import pandas as pd
import json
from utils import json_from_pandas_to_main_format
from scipy.special import softmax

class TextModel:
    def __init__(self, n_classes=371):
        self.train_data = pd.read_csv('All_cities_super_final_version.csv')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
        #
        for name, param in self.embedder.named_parameters():
            param.requires_grad = False
        #
        self.model = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes),
        )
        self.city_map = {0: ['Нижний Новгород', 'Ярославль', 'Екатеринбург', 'Владимир'],
                         1: ['Нижний Новгород'],
                         2: ['Ярославль'], 3: ['Екатеринбург'], 4: ['Владимир']}

        self.model.load_state_dict(torch.load("models/text_model.pt", map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.find_XID = pd.read_csv('find_XID.csv')
        self.find_XID.index = range(self.find_XID.shape[0])
        XIDS = self.find_XID['1'].to_dict()
        self.inverse_transform = {v:XIDS[i] for i,v in self.find_XID['0'].to_dict().items()}
        self.forward_transform = {v: k for k, v in self.inverse_transform.items()}
        print("model loaded")

    def get_prediction_embeding(self, query, city=None):
        query_embading = self.embedder.encode(query)
        if city:
            query_embading += self.embedder.encode(city)
        return query_embading

    def generate(self, root):
        self.train_data['Manhattan_distance'] = abs(self.train_data['Lon'] - root['Lon']) + abs(
            self.train_data['Lat'] - root['Lat'])
        sorted_df = self.train_data.sort_values(by='Manhattan_distance')

        nearest_places = sorted_df.iloc[:15]

        return nearest_places[['Lon', 'Lat']].values.tolist()

    def predict(self, text, city):
        select_city = {0: None, 1: 'Нижний Новгород', 2: 'Ярославль', 3: 'Екатеринбург', 4: 'Владимир'}
        city = select_city[city]
        query = torch.tensor(self.get_prediction_embeding(text, city))

        prob = nn.Softmax(dim=0)(self.model(query)).squeeze().cpu()
        data = self.get_topk(prob)
        dist = self.get_dist(data)
        gen = self.generate(data.iloc[data['score'].argmax()])
        return {'result': {'categories': [{'label': label, 'prob': prob} for label, prob in dist.items()],
                           'objects': json_from_pandas_to_main_format(
                               data.to_json(orient='records', force_ascii=False))},
                'route': gen}


    def get_dist(self, data):
        dist = {'жилье': 0, 'археологические музеи': 0, 'археология': 0, 'архитектура': 0, 'художественные галереи': 0,
                'банк': 0, 'банки': 0, 'биографические музеи': 0, 'мосты': 0, 'места захоронений': 0, 'замки': 0,
                'соборы': 0, 'католические церкви': 0, 'кладбища': 0, 'детские музеи': 0, 'детские театры': 0,
                'церкви': 0, 'кинотеатры': 0, 'цирки': 0, 'концертные залы': 0, 'культурный': 0, 'защитные стены': 0,
                'разрушенные объекты': 0, 'восточные ортодоксальные церкви': 0, 'музеи моды': 0, 'продукты питания': 0,
                'фортификационные сооружения': 0, 'укрепленные башни': 0, 'фонтаны': 0, 'сады и парки': 0,
                'геологические образования': 0, 'исторические': 0, 'историческая архитектура': 0,
                'исторические районы': 0, 'исторические дома музеи': 0, 'исторические объекты': 0,
                'исторические поселения': 0, 'исторические места': 0, 'исторические музеи': 0,
                'промышленные объекты': 0, 'инсталляции': 0, 'интересные места': 0, 'кремли': 0, 'местные музеи': 0,
                'усадьбы': 0, 'военные музеи': 0, 'монастыри': 0, 'памятники': 0, 'монументы и памятники': 0,
                'мечети': 0, 'горные вершины': 0, 'музеи': 0, 'музеи науки и технологии': 0, 'музыкальные места': 0,
                'национальные музеи': 0, 'природные': 0, 'природные монументы': 0, 'природные заповедники': 0,
                'музеи под открытым небом': 0, 'оперные театры': 0, 'другие': 0, 'другие археологические объекты': 0,
                'другие мосты': 0, 'другие здания и сооружения': 0, 'другие места захоронения': 0, 'другие церкви': 0,
                'другие отели': 0, 'другие музеи': 0, 'другие природоохранные зоны': 0,
                'другие технологические музеи': 0, 'другие храмы': 0, 'другие театры': 0, 'другие башни': 0,
                'планетарии': 0, 'кукольные театры': 0, 'железнодорожные станции': 0, 'религия': 0, 'рестораны': 0,
                'скальные образования': 0, 'научные музеи': 0, 'скульптуры': 0, 'спорт': 0, 'площади': 0, 'стадионы': 0,
                'театры и развлечения': 0, 'туристические объекты': 0, 'туристический объект': 0, 'башни': 0,
                'триумфальные арки': 0, 'неклассифицированные объекты': 0, 'городская среда': 0,
                'смотровые площадки': 0, 'военные памятные места': 0, 'водяные башни': 0, 'зоопарки': 0, 'школы': 0,
                'образование': 0, 'университеты': 0, 'тюрьма': 0}
        i = 0
        for _, line in data.iterrows():
            for kind in line['Kind'][1:-1].split(', '):
                dist[kind[1:-1]] += 1
                i += 1
        if i != 0:
            dist = {k: v / i for k, v in dist.items()}
        return dist

    def get_topk(self, pred, k=15):
        v, i = torch.topk(pred, k)
        vi = {i: v for i, v in zip(i.detach().numpy(), v.detach().numpy())}
        city_names = [self.inverse_transform[ind] for ind in i.numpy()]
        data = self.train_data.loc[self.train_data.XID.isin(city_names)] & (self.train_data.City.isin(city))
        scores = data['XID'].apply(lambda x: vi[self.forward_transform[x]])
        data['score'] = scores
        data = data.sort_values(by='score').loc[::-1]
        return data
