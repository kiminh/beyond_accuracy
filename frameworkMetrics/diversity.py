'''
Created on 12 Feb 2015

@author: mkaminskas
'''

from utils import config


def getListDiversity(training_data, item_list, method):
    '''
    list diversity, computed as defined by [McClave & Smyth, 2001]: diversity(R) = SUM{i in R} SUM{j in R\i} {dist(v_i, v_j)} / |R| * |R|-1
    where dist(v_i, v_j) is the distance between the items' vectors which can be computed using ratings or content labels
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_list: (id, score) tuples of items
    @type item_list: list
    
    @param method: the method used to compute list diversity {div_r, div_c}
    @type method: string
    
    @return: diversity of the item list
    @rtype: float
    '''
    
    diversity_sum = 0.0
    
    for i, (item_id, _) in enumerate(item_list[:-1]):
        tmp_item_list = [tup for j, tup in enumerate(item_list) if j > i]
        diversity_sum += _getItemDiversity(training_data, item_id, tmp_item_list, method, average=False)
    
    return 2 * diversity_sum / ( len(item_list) * (len(item_list)-1) )


def _getItemDiversity(training_data, item_id, item_list, method, average=True):
    '''
    item diversity score, as defined by [McClave & Smyth, 2001]: diversity(i,R) = SUM{j in R}{dist(v_i, v_j)} / |R|
    where dist(v_i, v_j) is the distance between the items' rating vectors [McClave & Smyth, 2001], or content label vectors [Ziegler et al. 2005]
    may be extended to include item factor vectors [Vargas and Castells, 2011], or binary rating vectors [Kelly & Bridge, 2006]
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_id: ID of item whose diversity we need to compute
    @type item_id: string
    
    @param item_list: the list of (id, score) tuples of items against which we compute the item's diversity 
    @type item_list: list
    
    @param method: the method used to compute item diversity {div_r, div_c}
    @type method: string
    
    @param average: the flag to disable averaging of the diversity sum (needed for getListDiversity method) 
    @type average: bool
    
    @return: diversity of the item with respect to the item_list
    @rtype: float
    '''
    
    if len(item_list) >= 1:
        item_index = training_data.getItemIndex(item_id)
        distance_sum = 0.0
        
        for i_id, _ in item_list:
            i_index = training_data.getItemIndex(i_id)
            
            if method == 'div_r':
                # need to convert the cosine similarity in [-1, 1] to distance in [0, 1]:
                # 1 - [(sim - min) / (max - min)] = (1 - sim) / 2
                distance_sum += (1.0 - training_data.item_similarity_matrix[item_index, i_index]) / 2.0
                
            elif method == 'div_c':
                a = set(config.ITEM_DATA[item_id]['labels'])
                b = set(config.ITEM_DATA[i_id]['labels'])
                c = a.intersection(b)
                distance_sum += 1 - ( float(len(c)) / (len(a) + len(b) - len(c)) )
                
#             elif method == 'diversity_factors':
#                 # need to divide the distance by 2.0 because there are negative values in vectors and therefore cosine ranges in [-1,1]
#                 distance_sum += spatial.distance.cosine(Q[:, item_index], Q[:, i_index]) / 2.0
#                 
#             elif method == 'diversity_ratings_bool':
#                 distance_sum += frameworkMetrics.getHammingDistance(training_data.matrix.getcol(item_index).indices, training_data.matrix.getcol(i_index).indices)
                
            else:
                raise ValueError('Wrong diversity computation method. Choose between div_r and div_c')
        
        if average:
            return distance_sum / len(item_list)
        else:
            return distance_sum
        
    else:
        return 1.0


# def getHammingDistance(vector_1, vector_2):
#     '''
#     calculate the (normalized) Hamming distance between two sparse vectors,
#     i.e., the number of positions where one vector has a non-zero value and the other doesn't, divided by the union of non-zero positions
#     '''
#     
#     v_1 = set(vector_1)
#     v_2 = set(vector_2)
#     
#     return float(len(v_1 | v_2) - len(v_1 & v_2)) / len(v_1 | v_2)

