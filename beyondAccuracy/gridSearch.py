'''
Created on 18 Aug 2015

grid search for WARP

@author: mkaminskas
'''

import logging
import os
from itertools import product

from mrec import load_fast_sparse_matrix
from mrec.mf.warp import WARPMFRecommender
from dataModel import trainData
from frameworkMetrics import topNLists
from dataHandling import dataPreprocessing
from utils import config


def parseResult():
    
    values = {}
    
    with open('../mus.txt','rb') as res_file:
        for line in res_file:
            l = line.replace('INFO gridSearch.<module>(): ','')
            if l.startswith('running fold '):
                l = l.rstrip('.\n')
                data = l.split(' with ')
                key = data[1]
                if key not in values:
                    values[key] = 0
            elif l.startswith('...done '):
                l = l.rstrip('.\n')
                data = l.split(' recall=')
                recall = float(data[1])
                assert key in values
                values[key] += recall
    
    for k,v in values.iteritems():
        print k,' = ',v/5
        

if __name__ == "__main__":
    
    parseResult()
    exit()
    
    config.MOVIES_OR_MUSIC = 'music'
    config.LOAD_DATA = 1
    config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, '../grid_search_music_splits/')
    config.LABEL_FREQUENCY_THRESHOLD = 10
    if not os.path.exists(config.SPLIT_DIR):
        os.makedirs(config.SPLIT_DIR)
    
    factor_values = [25, 50, 75, 100]
    C_values = [1.0, 10.0, 100.0, 1000.0]
    gamma_values = [0.01, 0.001, 0.0001]
    
    filenames = dataPreprocessing.loadData(mode='beyond_accuracy')
    # 5-fold cross-validation
    for iteration, (train_filename, test_filename, user_means_filename, eval_item_filename) in enumerate(filenames, 1):
        
        mrec_train_data = load_fast_sparse_matrix('tsv', train_filename)
        # create the training data and required recommendation models
        train_data = trainData.TrainData(train_filename, user_means_filename)
        
        for factor_value, C_value, gamma_value in product(factor_values, C_values, gamma_values):
            
            warp_recommender = WARPMFRecommender(d=factor_value, gamma=gamma_value, C=C_value)
            warp_recommender.fit(mrec_train_data.X)
            
            
            logging.info('running fold {0} with f={1}, C={2}, g={3}...'.format(iteration, factor_value, C_value, gamma_value))
            
            recall = 0
            evaluation_cases = 0
            with open(eval_item_filename,'rb') as eval_file:
                
                for line in eval_file:
                    data = line.split('\t')
                    user_id = data[0]
                    ground_truth_items = data[1].split(',')
                    random_unrated_items = data[2].rstrip('\n').split(',')
                    
                    # THIS IS A FIX FOR THE FEW USERS IN LAST.FM WHO HAVE IDENTICAL RATINGS FOR ALL TRAIN ITEMS
                    # WE HAVE TO SKIP THOSE USERS, BECAUSE THEY LOOSE ALL RATINGS IN THE MEAN-CENTERED MATRIX!
                    user_index = train_data.getUserIndex(user_id)
                    if len(train_data.getUserProfileByIndex(user_index)) < 1:
                        continue
                    
                    try:
                        evaluation_item_ids = ground_truth_items + random_unrated_items
                        rec_list_szie = config.RECOMMENDATION_LIST_SIZE * config.DIVERSIFICATION_CANDIDATES_FACTOR
                        predictions = warp_recommender.recommend_items(mrec_train_data.X, int(user_id)-config.MREC_INDEX_OFFSET, max_items=10000, return_scores=True)
                        top_recs = topNLists.getTopNList(predictions, rec_list_szie, evaluation_item_ids)
                        recall += topNLists.getRecall(ground_truth_items, top_recs)
                        evaluation_cases += 1
                    except Exception, e:
                        logging.info('couldave shouldave, but didnt: {0}'.format(e))
                    
                    if evaluation_cases == 300:
                        break
            
            logging.info('...done recall={0}'.format(float(recall) / evaluation_cases))
