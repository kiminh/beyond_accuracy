# '''
# Created on 22 Jan 2015
# 
# @author: mkaminskas
# '''
# 
# from dataModel.dataMatrix import DataMatrix
# from utils import config
# 
# 
# class TestData(DataMatrix):
#     '''
#     a class for creating the user-item test data as a CSR sparse matrix;
#     also implements operations for getting the 5-star user's items
#     '''
#     
#     
#     def __init__(self, data_path):
#         '''
#         read data from the split file, create ID-index dictionaries for rows (users) and columns (items)
#         create the csc sparse rating matrix
#         
#         @param data_path: path to a data split file
#         @type data_path: string
#         '''
#         
#         super(DataMatrix, self).__init__(data_path)
#     
#     
# #     def getUsers5StarItemsByIndex(self, user_index):
# #         '''
# #         @param user_index: index of the user
# #         @type user_index: string
# #         
# #         @return: IDs of the test items the user has rated above (or equal to) the RATING_THRESHOLD
# #         @rtype: list
# #         '''
# #         
# #         user_row = self.user_similarity_matrix.getrow(user_index)
# #         test_items = [i[0] for i in zip(user_row.indices, user_row.data) if i[1] >= config.RATING_THRESHOLD]
# #         
# #         return [self.getItemId(ind) for ind in test_items]
#         
#         