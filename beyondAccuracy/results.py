'''
Created on 25 Mar 2014

A class for storing the beyondAccuracy results of a single run
(either a single iteration over all users, or a run over N cross-validation folds)

In the first case the results are averaged across users (beyondAccuracy cases),
in the second case they are aggregated using addCVFold() method

rec_algorithm = {factor, user-based, item-based}
reranking_method = {div_c, div_r, sur_r, sur_c, nov}

@author: mkaminskas
'''

from beyondAccuracy import reranking
from frameworkMetrics import diversity, novelty, serendipity, topNLists
from utils import config


class Results(object):
    
    def __init__(self, rec_algorithm, reranking_method, alpha):
        
        self.rec_algorithm = rec_algorithm
        self.reranking_method = reranking_method
        self.alpha = alpha
        
        self.recall = 0.0
        self.diversity_c = 0.0
        self.diversity_r = 0.0
        self.surprise_c = 0.0
        self.surprise_r = 0.0
        self.surprise_r_n = 0.0
        self.novelty = 0.0
        self.overlap_w_baseline = 0.0
        self.item_set = set()
        self.file = None
    
    @property
    def coverage(self):
        return len(self.item_set)
    
    @property
    def result_string(self):
        return str(self.recall)+'\n'+str(self.diversity_c)+'\n'+str(self.diversity_r)+'\n'+str(self.surprise_c)+'\n'+ \
               str(self.surprise_r)+'\n'+str(self.surprise_r_n)+'\n'+str(self.novelty)+'\n'+str(self.overlap_w_baseline)+'\n'+str(self.coverage)
    
    
    def computePerformanceMetrics(self, user_index, predictions, ground_truth_items, data_matrix, options):
        '''
        update the attributes of results object with the performance topNLists for a list of item predictions
        returns a string of metric values to be written to raw data file
        '''
        
        R_baseline = predictions[:config.RECOMMENDATION_LIST_SIZE]
        
        if self.reranking_method:
            #----------- re-ranking ----------
            R_reranked = reranking.getRerankedRecommendations(data_matrix, self.reranking_method, predictions, self.alpha, user_index)
        else:
            R_reranked = R_baseline
        
        
        # ----------- beyondAccuracy ----------------------
        recall = topNLists.getRecall(ground_truth_items, R_reranked)
        self.recall += recall
        
        diversity_c = diversity.getListDiversity(data_matrix, R_reranked, 'div_c')
        self.diversity_c += diversity_c
        diversity_r = diversity.getListDiversity(data_matrix, R_reranked, 'div_r')
        self.diversity_r += diversity_r
        
        surprise_c = serendipity.getListSerendipity(data_matrix, user_index, R_reranked, 'sur_c')
        self.surprise_c += surprise_c
        surprise_r = serendipity.getListSerendipity(data_matrix, user_index, R_reranked, 'sur_r')
        self.surprise_r += surprise_r
#         surprise_r_n = serendipity.getListSerendipity(data_matrix, user_index, R_reranked, 'sur_r_n')
#         self.surprise_r_n += surprise_r_n
        
        nov = novelty.getListNovelty(data_matrix, R_reranked)
        self.novelty += nov
        
        overlap_w_baseline = topNLists.computeRecommendationListOverlap(R_baseline, R_reranked)
        self.overlap_w_baseline += overlap_w_baseline
        
        self.item_set.update([tup[0] for tup in R_reranked])
#         self.list_distance_w_baseline = topNLists.getListDistance(R_baseline, R_reranked)
        
        return str(data_matrix.getUserId(user_index))+':'+str(len(data_matrix.getUserProfileByIndex(user_index)))+':'+ \
               str(recall)+':'+str(diversity_c)+':'+str(diversity_r)+':'+ \
               str(surprise_c)+':'+str(surprise_r)+':'+str(-1.0)+':'+str(nov)+':'+str(overlap_w_baseline)+':predicted item = '+str([pred[0] for pred in R_reranked if pred[0] in ground_truth_items])+'\n'
    
    
    
    def averageMetricValues(self, evaluation_cases):
        '''
        average the result values across users (represented by num. of beyondAccuracy cases)
        '''
        
        self.recall /= evaluation_cases
        self.diversity_c /= evaluation_cases
        self.diversity_r /= evaluation_cases
        self.surprise_c /= evaluation_cases
        self.surprise_r /= evaluation_cases
        self.surprise_r_n /= evaluation_cases
        self.novelty /= evaluation_cases
        self.overlap_w_baseline /= evaluation_cases
    
    