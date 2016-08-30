# '''
# Created on 24 Feb 2014
# 
# A class containing information about a particular user: items he rated, his neighbours and their rated item indices.
# Needed for Serendipity computation.
# 
# @author: mkaminskas
# '''
# 
# import operator
# import numpy as np
# import sys
# import math
# 
# from beyondAccuracy import frameworkMetrics
# from dataHandling import dataReading
# from utils import config
# 
# 
# class UserProfile(object):
#     
#     def __init__(self, user_id, data_matrix, item_content, num_of_neighb):
#         
#         self.user_id = user_id
#         self.user_index = data_matrix.getUserIndex(user_id)
#         self.rated_items = list(data_matrix.getUserProfileByIndex(self.user_index))
#         self.neighbours_num = num_of_neighb
# #         self.neighbours = data_matrix.getUserNeighboursByIndex(self.user_index, num_of_neighb)
#         
#         # neighbourhood_item_popularity needed only if measuring serendipity in user's neighbourhood
# #         neighbour_items = []
# #         for neighbour_index, _ in self.neighbours:
# #             items = [tup[0] for tup in data_matrix.getUserProfileByIndex(neighbour_index)]
# #             neighbour_items += items
# #         item_pop = counter.Counter(neighbour_items)
# #         self.neighbourhood_item_popularity = sorted(item_pop.iteritems(), key=operator.itemgetter(1), reverse=False)
#     
#     
# #     def getNeighbourIndices(self):
# #         return [tup[0] for tup in self.neighbours]
#     
#     
#     
#     
#     def getLabelRarityInProfile(self, label, data_matrix, item_content):
#         count = 0.0
#         for item_index in self.rated_items:
#             item_id = data_matrix.getItemId(item_index)
#             if label in item_content[item_id]:
#                 count += 1.0
#         n = float(len(self.rated_items))
#         return 1.0 - (count / n)
#     
#     
#     def getUserIndexRarityInProfile(self, user_index, data_matrix):
#         count = 0.0
#         for item_index in self.getRatedItemIndices():
#             if user_index in data_matrix.getItemProfileByIndex(item_index):
#                 count += 1.0
#         n = float(len(self.getRatedItemIndices()))
#         return 1.0 - (count / n)
#     
#     
#     
#     
#     
#     '''
#     get the indices of 1000 random unrated items for a given user
#     '''
#     def get1000unratedItems(self, data_matrix, test_data):
#         
#         # get the indices of items in test dataReading rated by the target user
#         test_item_indices = [data_matrix.getItemIndex(d_tuple[1]) for d_tuple in test_data if d_tuple[0]==self.user_id]
#         
#         unrated_items_indices = [i for i in data_matrix.col_indices.values() if (i not in self.rated_items) and (i not in test_item_indices)]
#         np.random.shuffle(unrated_items_indices)
#         
#         return unrated_items_indices[:1000]
#     
#     
#     '''
#     get the dict of all items rated by the user: {item_id : {title : 'title', rating : R, labels: [label1, label2, ...], ...}
#     '''
#     def getProfile(self, data_matrix, item_content, dataset):
#         profile = []
#         
#         if dataset == 'movielens':
#             content_file = config.MOVIE_FILE_IMDB
#         elif dataset == 'lastfm':
#             content_file = config.ARTIST_FILE_LASTFM
#         
#         for item_ind in self.rated_items:
#             
#             item_id = data_matrix.getItemId(item_ind)
#             title = dataReading.getItemTitleById(item_id, content_file)
#             rating = data_matrix.original_matrix[self.user_index, item_ind]
#             profile.append( (rating, {'index': item_ind, 'title': title, 'labels': item_content[item_id]}) ) 
#         
#         return sorted(profile, key=operator.itemgetter(0), reverse=True)
#         
#         