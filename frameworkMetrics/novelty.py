'''
Created on 12 Feb 2015

@author: mkaminskas
'''
import math
import operator


def getListNovelty(training_data, item_list):
    '''
    the average item list novelty
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_list: (id, score) tuples of items
    @type item_list: list
    
    @return: novelty of the item list
    @rtype: float
    '''
    
    novelty_sum = 0.0
    
    for item_id, _ in item_list:
        novelty_sum += _getItemNovelty(training_data, item_id)
    
    return novelty_sum / len(item_list)


def getMinMaxNovelty(training_data, item_list):
    '''
    the min and max values of item list novelty
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_list: (id, score) tuples of items
    @type item_list: list
    
    @return: min and max novelties of the item list
    @rtype: tuple
    '''
    
    tmp_novelties = []
    for item_id, _ in item_list:
        tmp_novelties.append((item_id, _getItemNovelty(training_data, item_id)))
    
    sorted_novelties = sorted(tmp_novelties, key=operator.itemgetter(1))
    return sorted_novelties[0][1], sorted_novelties[-1][1]


def _getItemNovelty(training_data, item_id):
    '''
    novelty of an item, computed as an inverse rating frequency
    (also known as 'mean self information'): -log_2(#item_raters / #all_raters)
    
    to normalize into [0,1] we divide the novelty score by max novelty which is -log_2(1 / #all_raters)
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_id: ID of item whose novelty we need to compute
    @type item_id: string
    
    @return: novelty of the item
    @rtype: float
    '''
    
    item_index = training_data.getItemIndex(item_id)
    user_number = float(training_data.getTotalUserNumber())
    max_novelty = math.log(1.0 / user_number, 2)
    
    num_of_raters = training_data.getNumOfItemRatersByIndex(item_index)
    
    if num_of_raters == 0:
        raise ValueError('Item {0} has no raters!'.format(item_id))
    
    else:
        return math.log(num_of_raters / user_number, 2) / max_novelty

