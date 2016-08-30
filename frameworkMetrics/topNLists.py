'''
Created on Dec 2, 2013

a module for beyondAccuracy frameworkMetrics: recall, diversity, serendipity, novelty, etc.

@author: Marius
'''

import math
from scipy.stats import kendalltau, spearmanr

from utils import config


def getRecall(ground_truth_ids, predictions):
    '''
    compute recall as the ratio of ground truth items that made it into the recommendation list
    in case of 1 ground truth item, it's either a hit or no
    
    @param ground_truth_ids: ids of ground truth items
    @type ground_truth_ids: list
    
    @param predictions: (id, score) tuples of predicted items, sorted by score in decreasing order
    @type predictions: list
    
    @return: the hit ratio
    @rtype: float
    '''
    hits = 0.0
    
    for item_id in ground_truth_ids:
        if item_id in [pred[0] for pred in predictions]:
            hits += 1.0
    
    return hits / config.HIGHLY_RATED_ITEMS_NUM


def getTopNList(predictions, list_size=config.RECOMMENDATION_LIST_SIZE, evaluation_item_ids=None):
    '''
    make the list of top-N recommendations
    if evaluation_item_ids is not None, filter predictions to include only items in evaluation_item_ids and add the index offset
    
    @param predictions: (id, score) tuples of predicted items, sorted by score in decreasing order
    @type predictions: tuple
    
    @param list_size: the length of top-N list (defaults to config.RECOMMENDATION_LIST_SIZE)
    @type list_size: int
    
    @param evaluation_item_ids: ids of items for which predictions need to be made
    @type evaluation_item_ids: list
    
    @return: top-N list of predictions
    @rtype: list
    '''
    
    if evaluation_item_ids is not None:
        rec_list = [(str(tup[0]+config.MREC_INDEX_OFFSET), tup[1]) for tup in predictions if str(tup[0]+config.MREC_INDEX_OFFSET) in evaluation_item_ids]
    else:
        rec_list = predictions
    
    return rec_list[:list_size]


def computeRecommendationListOverlap(predictions1, predictions2):
    '''
    compute the normalized overlap between recommendation lists produced by 2 prediction algorithms
    
    @param predictions1: list of (item_id, prediction_score) tuples
    @type predictions1: list
    
    @param predictions2: list of (item_id, prediction_score) tuples
    @type predictions2: list
    
    @return: normalized overlap of the two lists
    @rtype: float
    '''
    
    return len(set([tup[0] for tup in predictions1]) & set([tup[0] for tup in predictions2])) / float(config.RECOMMENDATION_LIST_SIZE)


def computeRecommendationListDistance(predictions1, predictions2):
    '''
    compute the list similarity between recommendation lists produced by 2 prediction algorithms,
    as defined by [Fagin et al., Comparing top k lists, 2003]:
    sim(list1, list2) = sum_{i in I} |pos(i, list1) - pos(i, list2)| / |I|
    '''
    
    list1 = [tup[0] for tup in predictions1]
    list2 = [tup[0] for tup in predictions2]
    
    distance = 0.0
    # we're assuming both lists have the same length
    k = len(list1)
    
    for i in list1:
        if i in list2:
            distance += math.fabs(list1.index(i) - list2.index(i))
        else:
            distance += math.fabs(list1.index(i) - k)
    for j in list2:
        if j not in list1:
            distance += math.fabs(list2.index(j) - k)
    
    return distance / (k * (k + 1))


def computeRecommendationListCorrelation(predictions1, predictions2):
    '''
    compute the Spearman's and Kendal-Tau correlations between two item lists sorted by score
    '''
    
    list1 = [tup[0] for tup in predictions1]
    list2 = [tup[0] for tup in predictions2]
    
    corr_spearman = spearmanr(list1, list2)
    corr_kendall = kendalltau(list1, list2)
    
    return corr_spearman, corr_kendall


