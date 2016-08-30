'''
Created on 6 Dec 2015

@author: mkaminskas
'''
import operator

from frameworkMetrics import novelty, serendipity, explanationMetrics
from utils import config


class ExplanationResults(object):

    def __init__(self):
        
        self.file = None
        
        # AR-based metrics
        self.coverage = 0.0
        self.accuracy = 0.0
#         self.discounted_accuracy = 0.0
#         self.inv_accuracy = 0.0
#         self.lift = 0.0
#         self.odds_ratio = 0.0
#         self.conviction = 0.0
#         self.inv_conviction = 0.0
#         self.info_gain = 0.0
#         self.inv_info_gain = 0.0
        self.length = 0.0
        
        # other metrics
        self.avg_novelty = 0.0
        self.min_novelty = 0.0
        self.max_novelty = 0.0
        
        self.avg_surprise = 0.0
        self.min_surprise = 0.0
        self.max_surprise = 0.0
        
        self.avg_sim_to_rec = 0.0
        self.min_sim_to_rec = 0.0
        self.max_sim_to_rec = 0.0
        
        self.num_of_likes = 0.0
        self.num_of_dislikes = 0.0
        self.num_of_neutrals = 0.0
        
        self.overlap_w_original = 0.0
    
    @property
    def result_string(self):
        return str(self.coverage)+'\n'+str(self.accuracy)+'\n'+\
               str(self.length)+'\n'+\
               str(self.avg_novelty)+'\n'+str(self.min_novelty)+'\n'+str(self.max_novelty)+'\n'+\
               str(self.avg_surprise)+'\n'+str(self.min_surprise)+'\n'+str(self.max_surprise)+'\n'+\
               str(self.avg_sim_to_rec)+'\n'+str(self.min_sim_to_rec)+'\n'+str(self.max_sim_to_rec)+'\n'+\
               str(self.num_of_likes)+'\n'+str(self.num_of_dislikes)+'\n'+str(self.num_of_neutrals)+'\n'+str(self.overlap_w_original)
    
    
    def computePerformanceMetrics(self, user_index, item_id, data_matrix, options):
        
#         item_index = data_matrix.getItemIndex(item_id)
        
        rule, rule_metrics = data_matrix.generateExplanations(user_index, item_id, options.algorithm, extended_candidates=options.extendedCandidates,\
                                                              acc_filter=options.accuracyFilter, verbose=False)
        
        if rule:
            assert rule_metrics['coverage'] > 0.0 and rule_metrics['accuracy'] > 0.0
            
            self.coverage += rule_metrics['coverage']
            self.accuracy += rule_metrics['accuracy']
#             self.discounted_accuracy += rule_metrics['discounted_accuracy']
#             self.inv_accuracy += rule_metrics['inverted_accuracy']
#             self.lift += rule_metrics['lift']
#             self.odds_ratio += rule_metrics['odds_ratio']
#             self.conviction += rule_metrics['conviction']
#             self.inv_conviction += rule_metrics['inverted_conviction']
#             self.info_gain += rule_metrics['info_gain']
#             self.inv_info_gain += rule_metrics['inverted_info_gain']
            self.length += float(len(rule))
            
            avg_nov = novelty.getListNovelty(data_matrix, rule)
            min_nov, max_nov = novelty.getMinMaxNovelty(data_matrix, rule)
            self.avg_novelty += avg_nov
            self.min_novelty += min_nov
            self.max_novelty += max_nov
            
            avg_sur = serendipity.getListSerendipity(data_matrix, user_index, rule, 'sur_c', exclude_item_itself=True)
            min_sur, max_sur = serendipity.getMinMaxSerendipity(data_matrix, user_index, rule, 'sur_c', exclude_item_itself=True)
            self.avg_surprise += avg_sur
            self.min_surprise += min_sur
            self.max_surprise += max_sur
            
            # just lazy to put this into a metric module for now...
            tmp_sims = []
            rec_labels = set(config.ITEM_DATA[item_id]['labels'])
            for i_id, _ in rule: 
                rule_item_labels = set(config.ITEM_DATA[i_id]['labels'])
                label_intersection = rec_labels.intersection(rule_item_labels)
                tmp_sims.append((i_id, float(len(label_intersection)) / (len(rec_labels) + len(rule_item_labels) - len(label_intersection)) ))
            sorted_sims = sorted(tmp_sims, key=operator.itemgetter(1))
            avg_sim = sum([sim for _,sim in sorted_sims]) / float(len(sorted_sims))
            min_sim = sorted_sims[0][1]
            max_sim = sorted_sims[-1][1]
            self.avg_sim_to_rec += avg_sim
            self.min_sim_to_rec += min_sim
            self.max_sim_to_rec += max_sim
            
            likez, dislikez, neutralz = explanationMetrics.getNumOfLikesDislikes(rule)
            self.num_of_likes += likez
            self.num_of_dislikes += dislikez
            self.num_of_neutrals += neutralz
            
            original_rule, _ = data_matrix.generateExplanations(user_index, item_id, 'accuracy', extended_candidates=False, acc_filter=False, verbose=False)
            overlap_w_original = len(set(rule) & set(original_rule)) / float(len(rule))
            self.overlap_w_original += overlap_w_original
            
            return str(data_matrix.getUserId(user_index))+':'+str(len(data_matrix.getUserProfileByIndex(user_index)))+':'+str(item_id)+':'+str(rule)+':'+\
                   str(rule_metrics['coverage'])+':'+str(rule_metrics['accuracy'])+':'+\
                   str(len(rule))+':'+\
                   str(avg_nov)+':'+str(min_nov)+':'+str(max_nov)+':'+\
                   str(avg_sur)+':'+str(min_sur)+':'+str(max_sur)+':'+\
                   str(avg_sim)+':'+str(min_sim)+':'+str(max_sim)+':'+\
                   str(likez)+':'+str(dislikez)+':'+str(neutralz)+':'+str(overlap_w_original)+'\n'
            
        else:
            return str(data_matrix.getUserId(user_index))+':'+str(len(data_matrix.getUserProfileByIndex(user_index)))+':'+str(item_id)+':Cannot explain \n'
        
    
    def averageMetricValues(self, evaluation_cases):
        
        self.coverage /= evaluation_cases
        self.accuracy /= evaluation_cases
#         self.discounted_accuracy /= evaluation_cases
#         self.inv_accuracy /= evaluation_cases
#         self.lift /= evaluation_cases
#         self.odds_ratio /= evaluation_cases
#         self.conviction /= evaluation_cases
#         self.inv_conviction /= evaluation_cases
#         self.info_gain /= evaluation_cases
#         self.inv_info_gain /= evaluation_cases
        self.length /= evaluation_cases
        
        self.avg_novelty /= evaluation_cases
        self.min_novelty /= evaluation_cases
        self.max_novelty /= evaluation_cases
        
        self.avg_surprise /= evaluation_cases
        self.min_surprise /= evaluation_cases
        self.max_surprise /= evaluation_cases
        
        self.avg_sim_to_rec /= evaluation_cases
        self.min_sim_to_rec /= evaluation_cases
        self.max_sim_to_rec /= evaluation_cases
        
        self.num_of_dislikes /= evaluation_cases
        self.num_of_likes /= evaluation_cases
        self.num_of_neutrals /= evaluation_cases
        
        self.overlap_w_original /= evaluation_cases
        
        
        
        
        