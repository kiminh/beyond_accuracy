'''
Created on 12 Mar 2015

@author: mkaminskas
'''

import logging
import operator

from scipy import stats


def standartize(prediction_tuples):
    '''
    standartize the scores using the z-score normalization
    '''
    
    prediction_scores = [tup[1] for tup in prediction_tuples]
    
    if len(set(prediction_scores)) == 1:
        # if all values are the same, do not standardize
        return prediction_tuples
    else:
        standartized_scores = stats.zscore(prediction_scores)
        standartized_tuples = zip([tup[0] for tup in prediction_tuples], standartized_scores)
        
        if [i[0] for i in sorted(prediction_tuples, key=operator.itemgetter(1))] != [j[0] for j in sorted(standartized_tuples, key=operator.itemgetter(1))]:
            logging.info('score standardization mismatch: {0} vs {1}'.format([i[0] for i in sorted(prediction_tuples, key=operator.itemgetter(1))], [j[0] for j in sorted(standartized_tuples, key=operator.itemgetter(1))]))
        
        return standartized_tuples



if __name__ == "__main__":
    
    candidates = [(1, 0.5), (2, 0.5), (3, 0.51)]
    
    print standartize(candidates)