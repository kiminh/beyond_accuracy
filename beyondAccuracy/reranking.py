'''
Created on Oct 22, 2013

beyond-accuracy optimization using re-ranking techniques

@author: Marius
'''

import copy
import operator
import sys

from frameworkMetrics import diversity, novelty, serendipity
from utils import config, scoreNorm


def getRerankedRecommendations(training_data, method, candidates, alpha, user_index):
    '''
    the main method to generate a re-ranked list of recommendations from the candidate list using a specified method:
    - greedy re-ranking of N*b candidates [Smyth & McClave, 2001] based on rating vector similarity / content label similarity;
    - serendipity-based re-ranking;
    - novelty-based re-ranking.
    (- greedy re-ranking of N*b candidates [Kelly & Bridge, 2006] based on binary rating vectors)
    '''
    new_ranking = []
    
    assert len(candidates) == config.RECOMMENDATION_LIST_SIZE * config.DIVERSIFICATION_CANDIDATES_FACTOR
    
    standartized_candidates = scoreNorm.standartize(candidates)
    
    
    if method == 'div_c' or method == 'div_r':
        # need to make a deep copy of candidates, because greedy algorithm changes the candidate list 
        topNb = copy.deepcopy(standartized_candidates)
        new_ranking = _greedyDiversification(training_data, topNb, alpha, method)
    
    else:
        new_ranking = _notSoGreedyDiversification(training_data, standartized_candidates, alpha, method, user_index)
    
    return new_ranking


def _greedyDiversification(training_data, candidates, alpha, method, verbose=False):
    '''
    implements the greedy diversification as first described by [McClave & Smyth, 2001]:
    given a candidate list C (of size N*b), construct a new list R of size N
     - compute the score for each item in C as
       (1 - alpha) * predictionScore + alpha * utilityScore,
       where utilityScore is the average item diversity w.r.t the other items in R, or novelty of the item
     - move the highest-scored item from C to R
     - repeat until the size of R reaches N
    method = {factors | ratings | ratings_bool | content}
    '''
    
    R = []
    
    while (len(R) < config.RECOMMENDATION_LIST_SIZE) and candidates:
        
        # compute diversity values of all candidate items, then standardize them
        candidate_diversities = []
        for item_id, _ in candidates:
            candidate_diversities.append( (item_id, diversity._getItemDiversity(training_data, item_id, R, method)) )
        
        standartized_diversities = scoreNorm.standartize(candidate_diversities)
        
        # identify the item with best combined score
        best_item = (-1, 0.0)
        for (item_id, diversity_score), (item_id_2, prediction_score) in zip(standartized_diversities, candidates):
            
            assert item_id == item_id_2
            item_score = ((1 - alpha) * prediction_score) + (alpha * diversity_score)
            
            if verbose:
                sys.stdout.write('\n\t (1 - '+str(alpha)+') * '+str(prediction_score)+' + '+str(alpha)+' * '+str(diversity_score)+' = '+str(item_score))
            
            if item_score > best_item[1]:
                best_item = (item_id, item_score)
        
        R.append(best_item)
        
        # remove the identified best item from C
        for item, pred in candidates:
            if item == best_item[0]:
                best_item = (item,pred)
                break
        
        try:
            candidates.remove(best_item)
        except ValueError:
            print 'Could not remove ',best_item,' from candidate list.'
            continue
    
    return sorted(R, key=operator.itemgetter(1), reverse=True)


def _notSoGreedyDiversification(training_data, candidates, alpha, method, user_index, verbose=False):
    '''
    a simpler version of re-ranking where item score is a combination of prediction value and novelty or surprise OR Explanation Metric:
    nov(i) = -log_2 p(item_rated)
    OR
    surprise(i) = min dist(U_profile, i)
    
    compared to greedy diversification, here we don't need to compute item's score wrt previously selected items
    '''
    
    R = []
    
    if method == 'sur_r_n':
        max_freq = training_data.getMaxItemCooccurrenceValue(candidates, user_index)
    else:
        max_freq = None
    
    # compute metric values of all candidate items, then standardize them
    metric_scores = []
    for item_id, _ in candidates:
        
        if method == 'nov':
            metric_score = novelty._getItemNovelty(training_data, item_id)
        
        elif method == 'sur_c':
            metric_score = serendipity._getItemSerendipity(training_data, user_index, item_id, max_freq, 'sur_c')
        
        elif method == 'sur_r':
            metric_score = serendipity._getItemSerendipity(training_data, user_index, item_id, max_freq, 'sur_r')
        
        elif method == 'sur_r_n':
            metric_score = serendipity._getItemSerendipity(training_data, user_index, item_id, max_freq, 'sur_r_n')
        
        
        elif 'expl' in method:
            
            rule_algorithm = None
            if method == 'expl_accuracy':
                rule_algorithm = 'accuracy'
            elif method == 'expl_discounted':
                rule_algorithm = 'discounted_accuracy_thresh'
                config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = 0
            elif method == 'expl_discounted_better':
                rule_algorithm = 'discounted_accuracy_thresh'
                config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = 1
            
            if rule_algorithm not in config.RULE_METRICS:
                config.RULE_METRICS.append(rule_algorithm)
            
            _, rule_metrics = training_data.generateExplanations(user_index, item_id, rule_algorithm, extended_candidates=1, acc_filter=1, verbose=False)
            
            if rule_metrics is None:
                metric_score = 0.0
            else:
                metric_score = rule_metrics[rule_algorithm]
            assert metric_score >= 0.0 
            
        else:
            raise ValueError('Wrong re-ranking method. Choose between nov, sur_c, sur_r, expl_accuracy, expl_discounted, and expl_discounted_better')
        
        metric_scores.append( (item_id, metric_score) )
    
    
    standartized_scores = scoreNorm.standartize(metric_scores)
    
    
    for (item_id, metric_score), (item_id_2, prediction_score) in zip(standartized_scores, candidates):
        
        assert item_id == item_id_2
        item_score = ((1 - alpha) * prediction_score) + (alpha * metric_score)
        if verbose:
            sys.stdout.write('\n\t (1 - '+str(alpha)+') * '+str(prediction_score)+' + '+str(alpha)+' x '+str(metric_score)+' = '+str(item_score))
        R.append((item_id, item_score))
        
    return sorted(R, key=operator.itemgetter(1), reverse=True)[:config.RECOMMENDATION_LIST_SIZE]


# def adomaviciusDiversification(predictions, item_popularity, alpha):
#     '''
#     implements popularity-based re-ranking as described by [Adomavicius and Kwon, 2012]
#     given an item i, it's rank is computed as:
#         rank_popularity(i), if prediction(i) > threshold
#         alpha + rank_standard(i), otherwise.
#     alpha is max of rank_popularity(k) for all items k whose prediction is > threshold.
#     i.e., alpha guarantees that all items ranked using standard ranking will appear after those ranked using popularity
#     
#     since we do not predict ratings, but only ranking, instead of the rating threshold, we use a percentage of ranked items,
#     i.e., threshold = 0.9 means top 90% of top N*b items are re-ranked according to popularity
#     '''
#     
#     ranked_items = []
#     item_cutoff_position = int( round(len(predictions) * alpha) )
#     
#     max_rank = 0
#     # first rank the items above the threshold
#     for item_index, _ in predictions[:item_cutoff_position]:
#         if item_index in [tup[0] for tup in item_popularity]:
#             item_rank = [tup[0] for tup in item_popularity].index(item_index) + 1
#             ranked_items.append((item_index, item_rank))
#             if item_rank > max_rank:
#                 max_rank = item_rank
#         # if an item is not in item_popularity, it has no ratings in training data
#         else:
#             ranked_items.append((item_index, 1))
#     
#     # then check for items below rating threshold
#     for index, (item_index, _) in enumerate(predictions[item_cutoff_position:]):
#         item_rank = max_rank + index + 1
#         ranked_items.append((item_index, item_rank))
#     
#     return sorted(ranked_items, key=operator.itemgetter(1), reverse=False)


if __name__ == "__main__":
    
#     print adomaviciusDiversification( ((1,0),(2,0),(3,0),(4,0),(5,0)), ((5,1),(4,10),(3,100),(2,1000),(1,10000)), 0.0 )
    pass
    