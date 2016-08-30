'''
Created on 22 Feb 2016

@author: mkaminskas
'''

import logging
import math
import random
import sys

from utils import config


def getRuleMetrics(data_matrix, antecedent, consequent, verbose):
        '''
        antecedent is a list of (item_ID, opinion) tuples and consequent is a single (item_id, opinion='like') tuple
        config.ITEM_OPINIONS is a dict storing (item_ID ,opinion) pairs and corresponding user_ID lists:
        '''
        
        rule_metrics = {}
        for m in config.RULE_METRICS:
            rule_metrics[m] = 0.0
        
        if verbose:
            logging.info('computing the metrics of {0} -> {1}'.format(antecedent, consequent))
        
        # COVERAGE
        list_of_raters = []
        for item_id, opinion in antecedent:
            item_raters = set(config.ITEM_OPINIONS[item_id, opinion])
            list_of_raters.append(item_raters)
        
        antecedent_raters = set.intersection(*list_of_raters)
        antecedent_raters_num = len(antecedent_raters)
        
        rule_metrics['coverage'] = _getRuleCoverage(antecedent_raters_num, data_matrix.getTotalUserNumber())
#         if verbose:
#             logging.info('there are {0} users who agree on {1}'.format(antecedent_raters_num, antecedent))
#             logging.info('and so the coverage is {0}/{1}={2}'.format(antecedent_raters_num, data_matrix.getTotalUserNumber(), rule_metrics['coverage']))
        
        if rule_metrics['coverage'] == 0:
            return rule_metrics
        
        
        # ACCURACY
        consequent_raters = set(config.ITEM_OPINIONS[consequent])
        # add to the list of antecedent raters, so in the end we get the intersection of all
        list_of_raters.append(consequent_raters)
        rule_raters = set.intersection(*list_of_raters)
        rule_raters_num = len(rule_raters)
        
        rule_metrics['accuracy'] = _getRuleAccuracy(rule_raters_num, antecedent_raters_num)
#         if verbose:
#             logging.info('there are {0} users who agree on {1}'.format(rule_raters_num, antecedent+[consequent]))
#             logging.info('and so the ACCURACY is {0}/{1}={2}'.format(rule_raters_num, antecedent_raters_num, rule_metrics['accuracy']))
        
        if rule_metrics['accuracy'] == 0:
            return rule_metrics
        
        if 'discounted_accuracy' in config.RULE_METRICS:
            
            # DISCOUNTED ACCURACY
            rule_metrics['discounted_accuracy'] = _getRuleDiscountedAccuracy(data_matrix, rule_metrics['accuracy'],\
                                                                             antecedent, consequent, antecedent_raters, verbose)
#             if verbose:
#                 logging.info('while the DISCOUNTED ACCURACY is {0}'.format(rule_metrics['discounted_accuracy']))
        
        if 'discounted_accuracy_thresh' in config.RULE_METRICS:
            
            # THRESHOLD-DISCOUNTED ACCURACY
            discount_by_better_explanations = config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS
            rule_metrics['discounted_accuracy_thresh'] = _getRuleThresholdDiscountedAccuracy(data_matrix, rule_metrics['accuracy'],\
                                                                             antecedent, consequent, antecedent_raters, discount_by_better_explanations, verbose)
        
        
        
        if any(i in config.RULE_METRICS for i in ['conviction', 'inverted_conviction', 'info_gain', 'inverted_info_gain']):
            
            ############# AR-based metrics ################
            
            all_raters = set(data_matrix._row_indices.keys())
            
    #         # ADDED VALUE
    #         if 'added_value' in rule_metrics:
    #             rule_metrics['added_value'] = _getRuleAddedValue(rule_raters_num, antecedent_raters_num, len(consequent_raters), data_matrix.getTotalUserNumber())
    #         # INVERTED ADDED VALUE
    #         if 'inverted_added_value' in rule_metrics:
    #             rule_metrics['inverted_added_value'] = _getRuleAddedValue(rule_raters_num, len(consequent_raters), antecedent_raters_num, data_matrix.getTotalUserNumber())
    #         # LEVERAGE
    #         if 'leverage' in rule_metrics:
    #             rule_metrics['leverage'] = _getRuleLeverage(rule_raters_num, antecedent_raters_num, len(consequent_raters), data_matrix.getTotalUserNumber())
    #         # INVERTED LEVERAGE
    #         if 'inverted_leverage' in rule_metrics:
    #             rule_metrics['inverted_leverage'] = _getRuleLeverage(rule_raters_num, len(consequent_raters), antecedent_raters_num, data_matrix.getTotalUserNumber())
    #         # ODDS RATIO
    #         if 'odds_ratio' in rule_metrics:
    #             rule_metrics['odds_ratio'] = _getRuleOddsRatio(ant_not_cons_freq, len(consequent_raters), rule_raters_num, data_matrix.getTotalUserNumber())
            
            # users who have the antecedent, but not the consequent in their profiles
            ant_not_cons_freq = len(antecedent_raters - consequent_raters)
            # users who have the consequent, but not the antecedent
            not_ant_cons_freq = len(consequent_raters - antecedent_raters)
            # users who have neither antecedent not consequent
            not_ant_not_cons_freq = len(all_raters - antecedent_raters - consequent_raters)
            # users who don't have antecedent
            not_ant_freq = len(all_raters - antecedent_raters)
            # users who don't have consequent
            not_cons_freq = len(all_raters - consequent_raters)
             
            # CONVICTION
            if 'conviction' in config.RULE_METRICS:
                rule_metrics['conviction'] = _getRuleConviction(ant_not_cons_freq, antecedent_raters_num, len(consequent_raters), data_matrix.getTotalUserNumber())
#                 if verbose:
#                     logging.info('there are {0} users who agree on {1} and NOT{2}'.format(ant_not_cons_freq,\
#                                                                                           [(data_matrix.getItemId(op[0]),op[1]) for op in antecedent],\
#                                                                                           (data_matrix.getItemId(consequent[0]),consequent[1])))
#                     logging.info('and so the CONVICTION is {0}*{1}/{2}={3}'.format(antecedent_raters_num, data_matrix.getTotalUserNumber()-len(consequent_raters),\
#                                                                                    ant_not_cons_freq, rule_metrics['conviction']))
            
            # INVERTED CONVICTION
            if 'inverted_conviction' in config.RULE_METRICS:
                rule_metrics['inverted_conviction'] = _getRuleConviction(not_ant_cons_freq, len(consequent_raters), antecedent_raters_num, data_matrix.getTotalUserNumber())
#                 if verbose:
#                     logging.info('there are {0} users who agree on NOT{1} and {2}'.format(not_ant_cons_freq,\
#                                                                                           [(data_matrix.getItemId(op[0]),op[1]) for op in antecedent],\
#                                                                                           (data_matrix.getItemId(consequent[0]),consequent[1])))
#                     logging.info('and so the INVERTED CONVICTION is {0}*{1}/{2}={3}'.format(data_matrix.getTotalUserNumber()-antecedent_raters_num, len(consequent_raters),\
#                                                                                             not_ant_cons_freq, rule_metrics['inverted_conviction']))
             
            # INFO GAIN
            if 'info_gain' in config.RULE_METRICS:
                rule_metrics['info_gain'] = _getRuleInfoGain(antecedent_raters_num, not_ant_freq, len(consequent_raters),\
                                                             not_cons_freq, data_matrix.getTotalUserNumber(), rule_raters_num, ant_not_cons_freq,\
                                                             not_ant_cons_freq, not_ant_not_cons_freq)
                assert (rule_metrics['info_gain'] >= 0) and (rule_metrics['info_gain'] <=1)
            
            # INVERTED INFO GAIN
            if 'inverted_info_gain' in config.RULE_METRICS:
                rule_metrics['inverted_info_gain'] = _getRuleInfoGain(len(consequent_raters), not_cons_freq, antecedent_raters_num,\
                                                                      not_ant_freq, data_matrix.getTotalUserNumber(), rule_raters_num, not_ant_cons_freq,\
                                                                      ant_not_cons_freq, not_ant_not_cons_freq)
                assert (rule_metrics['inverted_info_gain'] >= 0) and (rule_metrics['inverted_info_gain'] <=1)
        
        return rule_metrics
    
    
def _getRuleCoverage(antecedent_freq, total_freq):
    '''
    coverage(X->Y) = P(X)
    '''
    return antecedent_freq / float(total_freq)
    

def _getRuleAccuracy(rule_freq, antecedent_freq):
    '''
    accuracy(X->Y) = P(Y | X)
    '''
    if antecedent_freq:
        return rule_freq / float(antecedent_freq)
    else:
        return 0.0


def _getRuleAddedValue(rule_freq, antecedent_freq, consequent_freq, total_freq):
    '''
    added_value(X->Y) = P(Y | X) - P(Y)
    '''
    part_1 = rule_freq / float(antecedent_freq)
    part_2 = consequent_freq / float(total_freq)
    
    return part_1 - part_2


def _getRuleLeverage(rule_freq, antecedent_freq, consequent_freq, total_freq):
    '''
    added_value(X->Y) = P(Y | X) - P(X)P(Y)
    '''
    part_1 = rule_freq / float(antecedent_freq)
    part_2 = (antecedent_freq*consequent_freq) / float(total_freq*total_freq)
    
    return part_1 - part_2


def _getRuleConviction(antecedent_and_not_consequent_freq, antecedent_freq, consequent_freq, total_freq):
    '''
    odds_ratio(X->Y) = P(X)P(not Y) / P(X | not Y)
    '''
    not_consequent_freq = float(total_freq - consequent_freq)
    
    numerator = (antecedent_freq*not_consequent_freq) / float(total_freq*total_freq)
    denominator = antecedent_and_not_consequent_freq / not_consequent_freq
    
    if denominator:
        return numerator / denominator
    else:
        return 0.0


def _getRuleOddsRatio(antecedent_and_not_consequent_freq, consequent_freq, rule_freq, total_freq):
    '''
    odds_ratio(X->Y) = P(X | Y) / P(X | not Y)
    '''
    not_consequent_freq = float(total_freq - consequent_freq)
    
    numerator = rule_freq / float(consequent_freq)
    denominator = antecedent_and_not_consequent_freq / not_consequent_freq
    
    if denominator:
        return numerator / denominator
    else:
        return 0.0


def _getRuleInfoGain(ant_freq, not_ant_freq, cons_freq, not_cons_freq, total_freq,\
                     ant_cons_freq, ant_not_cons_freq, not_ant_cons_freq, not_ant_not_cons_freq):
    '''
    info_gain(X->Y) = ( P(X ^ Y) * log [ P(X ^ Y)/P(X)P(Y) ] + 
                        P(X ^ not Y) * log [ P(X ^ not Y)/P(X)P(not Y) ] +
                        P(not X ^ Y) * log [ P(not X ^ Y)/P(not X)P(Y) ] +
                        P(not X ^ not Y) * log [ P(not X ^ not Y)/P(not X)P(not Y) ] )
                    / ( P(X) * log P(X) + P(not X) * log P(not X) ) 
                      
    '''
    
    if any(v == 0.0 for v in [ant_freq, not_ant_freq, cons_freq, not_cons_freq, ant_cons_freq, ant_not_cons_freq, not_ant_cons_freq, not_ant_not_cons_freq]):
        return 0.0
    
    numerator = ant_cons_freq/float(total_freq) * math.log( (total_freq * ant_cons_freq) / (ant_freq * float(cons_freq)), 2 ) + \
                ant_not_cons_freq/float(total_freq) * math.log( (total_freq * ant_not_cons_freq) / (ant_freq * float(not_cons_freq)), 2 ) + \
                not_ant_cons_freq/float(total_freq) * math.log( (total_freq * not_ant_cons_freq) / (not_ant_freq * float(cons_freq)), 2 ) + \
                not_ant_not_cons_freq/float(total_freq) * math.log( (total_freq * not_ant_not_cons_freq) / (not_ant_freq * float(not_cons_freq)), 2 )
    
    denominator = ant_freq/float(total_freq) * math.log(ant_freq/float(total_freq), 2) + \
                  not_ant_freq/float(total_freq) * math.log(not_ant_freq/float(total_freq), 2)
    
    if denominator != 0.0:
        return numerator / -denominator  
    else:
        return 0.0


def _getRuleDiscountedAccuracy(data_matrix, target_accuracy, antecedent, consequent, antecedent_raters, verbose):
    
    # store rule as set of item IDs
    rule_item_ids =  set([r[0] for r in antecedent+[consequent]])
    
    if verbose:
        logging.info('\t checking the potential accuracy for rule items {0} minus the consequent {1}'.format(rule_item_ids, consequent[0]))
    
    # for each other item in the dataset (or a sample of those items), accumulate accuracy of the potential rules
    other_accuracies = 0.0
    counter = 0
    
    gen = (raters for opinion, raters in config.ITEM_OPINIONS.iteritems() if (opinion[1]=='like') and (opinion[0] not in rule_item_ids) and (any(a in raters for a in antecedent_raters)) )
    for raters in gen:
        counter += 1
        potential_rule_raters = set.intersection(antecedent_raters, raters)
        other_accuracies += _getRuleAccuracy(len(potential_rule_raters), float(len(antecedent_raters)))
        
#         print counter,raters,len(potential_rule_raters),'/',len(antecedent_raters),_getRuleAccuracy(len(potential_rule_raters), float(len(antecedent_raters)))
#     print counter, other_accuracies
#     assert counter <= data_matrix.getTotalItemNumber() - len(rule_item_ids)
    
    if verbose:
        logging.info('\t got {0} potential rules, which gives a discounted accuracy of {1}'.format(counter, target_accuracy / (other_accuracies + 1.0)))
    
    return target_accuracy / (other_accuracies + 1.0)



def _getRuleThresholdDiscountedAccuracy(data_matrix, target_accuracy, antecedent, consequent, antecedent_raters, discount_by_better_explanations, verbose):
    
    # store rule as set of item IDs
    rule_item_ids =  set([r[0] for r in antecedent+[consequent]])
    
    if verbose:
        logging.info('\t checking the potential accuracy for rule items {0} minus the consequent {1}'.format(rule_item_ids, consequent[0]))
    
    # for each other item in the dataset, check accuracy of the potential rules
    denominator = 0.0
    gen = (raters for opinion, raters in config.ITEM_OPINIONS.iteritems() if (opinion[1]=='like') and (opinion[0] not in rule_item_ids) and (any(a in raters for a in antecedent_raters)) )
    
    for raters in gen:
        if discount_by_better_explanations:
            potential_rule_raters = set.intersection(antecedent_raters, raters)
            if _getRuleAccuracy(len(potential_rule_raters), float(len(antecedent_raters))) >= target_accuracy:
                denominator += 1.0
        else:
            denominator += 1.0
    
    if verbose:
        better = ''
        if discount_by_better_explanations:
            better = '(i.e., better)'
        logging.info('\t got {0} potential{1} rules, which gives a discounted accuracy of {2}'.format(denominator, better, target_accuracy / (denominator + 1.0)))
    
    return target_accuracy / (denominator + 1.0)


    
    
def getNumOfLikesDislikes(antecedent):
    '''
    get the number of liked/disliked/neutral items in the antecedent
    '''
    likes = 0
    dislikes = 0
    neutrals = 0
    
    for _, opinion in antecedent:
        if opinion == 'like':
            likes += 1
        elif opinion == 'dislike':
            dislikes += 1
        elif opinion == 'neutral':
            neutrals += 1
        else:
            raise ValueError('Wrong opinion literal encountered:',opinion)
    
    return likes, dislikes, neutrals


