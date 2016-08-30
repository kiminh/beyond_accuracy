'''
Created on 22 Jan 2015

@author: mkaminskas
'''
import logging
import operator
import scipy.sparse as sp

from utils import counter


class DataMatrix(object):
    '''
    a base class for the user-item rating matrix;
    the rating data is stored as CSR sparse matrix;
    users/items are stored as rows/columns respectively
    '''
    
    def __init__(self, data_path):
        '''
        read (user,item,rating) tuples from the data split file,
        create ID-index dictionaries for rows (users) and columns (items),
        create the CSR sparse rating matrix
        
        @param data_path: path to a data split file
        @type data_path: string
        '''
        
        logging.info('creating the rating matrix from {0}...'.format(data_path))
        
        data_tuples = []
        with open(data_path, 'rb') as data_file:
            for line in data_file:
                data = line.split('\t')
                
                user_id = data[0]
                item_id = data[1]
                rating = float(data[2].rstrip('\r\n'))
                
                data_tuples.append((user_id, item_id, rating))
        
        self._row_indices = {}
        all_users = set(d_tuple[0] for d_tuple in data_tuples)
        for user_index, user_id in enumerate(sorted(list(all_users))):
            self._row_indices[user_id] = user_index
        
        self._col_indices = {}
        all_items = set(d_tuple[1] for d_tuple in data_tuples)
        for item_index, item_id in enumerate(sorted(list(all_items))):
            self._col_indices[item_id] = item_index
        
        self.__item_popularity = counter.Counter(d_tuple[1] for d_tuple in data_tuples)
        
        data_matrix = sp.lil_matrix((len(self._row_indices), len(self._col_indices)))
        for user_id, item_id, rating in data_tuples:
            data_matrix[self._row_indices[user_id], self._col_indices[item_id]] = rating
        self.rating_matrix = data_matrix.tocsr()
        
        logging.info('done!')
    
    
    def getItemIndex(self, item_id):
        '''
        @param item_id: ID of the item
        @type item_id: string
        '''
        return self._col_indices[item_id]
    
    
    def getItemId(self, item_index):
        '''
        @param item_index: index of the item
        @type item_index: string
        '''
        for item_id, index in self._col_indices.iteritems():
            if index == item_index:
                return item_id
        return -1
    
    
    def getUserIndex(self, user_id):
        '''
        @param user_id: ID of the user
        @type user_id: string
        '''
        return self._row_indices[user_id]
    
    
    def getUserId(self, user_index):
        '''
        @param user_index: index of the user
        @type user_index: string
        '''
        for user_id, index in self._row_indices.iteritems():
            if index == user_index:
                return user_id
        return -1
    
    
    def getPopularityInfo(self):
        '''
        @return: the list of tuples (item_id, num_of_ratings) sorted from the least rated
        @rtype list
        '''
        return sorted(self.__item_popularity.iteritems(), key=operator.itemgetter(1), reverse=False)
    
    
    def getPopularityDict(self):
        '''
        @return: the dict {item_id: num_of_ratings}
        @rtype dict
        '''
        return self.__item_popularity
    
    
    def getTotalUserNumber(self):
        return self.rating_matrix.shape[0]
    
    
    def getTotalItemNumber(self):
        return self.rating_matrix.shape[1]
    
    
    def getNumOfItemRatersByIndex(self, item_index):
        '''
        @param index of the item
        @type string
        '''
        return self.rating_matrix.getcol(item_index).getnnz()
    
    
    def getUserProfileByIndex(self, user_index, filter_by_rating = None):
        '''
        @return: array of item indices rated by the user (possibly only of items with the rating value 'filter_by_rating')
        @rtype array
        '''
        user_row = self.rating_matrix.getrow(user_index)
        
        if filter_by_rating:
            filtered_item_indices = []
            user_id = self.getUserId(user_index)
            for item_index in user_row.indices:
                if round(self._user_means[user_id] + self.rating_matrix[user_index, item_index], 1) == filter_by_rating:
                    filtered_item_indices.append(item_index)
            return filtered_item_indices
            
        else:
            return user_row.indices
    
    
    def getItemProfileByIndex(self, item_index, user_indices=None):
        '''
        @return: array of get item raters' indices (possibly only among the specified users)
        @rtype array
        '''
        item_column = self.rating_matrix.getcol(item_index).tocsc()
        
        if user_indices:
            return [i for i in item_column.indices if i in user_indices]
        else:
            return item_column.indices    
    
    
    def getMaxItemCooccurrenceValue(self, item_list, user_index):
        
        profile_items = self.getUserProfileByIndex(user_index)
        candidate_items = [self.getItemIndex(item_id) for item_id, _ in item_list]
        items = set(profile_items) | set(candidate_items)
        
        freqs = []
        
        for item_index1 in items:
            for item_index2 in items:
                if item_index1 != item_index2:
                    raters_x = set(self.getItemProfileByIndex(item_index1))
                    raters_y = set(self.getItemProfileByIndex(item_index2))
                    freqs.append(float(len(raters_x & raters_y)))
        
        return max(freqs)
    
    