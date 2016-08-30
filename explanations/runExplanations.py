'''
Created on 4 Dec 2015

@author: mkaminskas
'''

import logging
import optparse
import os
import pickle
import random
import sys

from dataHandling import dataPreprocessing
from dataModel import trainData
from explanations import explanationResults
from utils import config


if __name__ == "__main__":
    
    usage = "runExplanations.py -d < movies | music > \
                         -l < 0 | 1 > \
                         -f < folder > \
                         -k <10 - 100> \
                         -t <1 - 10> \
                         -s < 0 | 1 > \
                         -o < 0 | 1 > \
                         -r < 0 | 1 > \
                         -e < 0 | 1 >"
    
    parser = optparse.OptionParser(usage)
    parser.add_option("-d", "--dataset", action="store", dest="dataset", help="the dataset to be used")
    parser.add_option("-l", "--loadData", type="int", action="store", dest="loadData", help="0/1 switch to generate splits or use existing ones", default=0)
    parser.add_option("-f", "--folder", action="store", dest="folder", help="the folder to store CV splits and beyondAccuracy")
    parser.add_option("-k", "--ksize", type="int", action="store", dest="ksize", help="the size of neighbourhood in knn approaches", default=150)
    parser.add_option("-t", "--threshold", type="int", action="store", dest="threshold", help="the label frequency threshold for movie data", default=10)
#     parser.add_option("-s", "--sample", type="int", action="store", dest="sample", help="the number of good/bad items to use (default=1000)", default=1000)
    parser.add_option("-o", "--opinions", type="int", action="store", dest="opinions", help="0/1 switch for loading the item opinions", default=0)
    parser.add_option("-e", "--extendedCandidates", type="int", action="store", dest="extendedCandidates", help="0/1 switch for the extended candidate opinion search", default=0)
    parser.add_option("-m", "--meanCenter", type="int", action="store", dest="meanCenter", help="0/1 switch for item opinion discretization", default=0)
    parser.add_option("-a", "--algorithm", action="store", dest="algorithm", help="the algorithm to use for explanation generation")
#     parser.add_option("-g", "--sampling", type="int", action="store", dest="sampling", help="the percentage of items to sample for discounted accuracy computation (default=0)", default=0)
    parser.add_option("-y", "--accuracyFilter", type="int", action="store", dest="accuracyFilter", help="0/1 switch for making discounted accuracy rules shorter (default=0)", default=0)
    parser.add_option("-w", "--discountThreshold", type="int", action="store", dest="discountThreshold", help="0/1 switch for discounted accuracy using num of better explanations (default=0)", default=0)
    (options, args) = parser.parse_args()
    
    if not options.dataset:
        parser.error("the required options not entered")
    
    
    config.MOVIES_OR_MUSIC = options.dataset
    config.LOAD_DATA = options.loadData
    config.LOAD_OPINIONS = options.opinions
    config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, '../explanatoin_data/'+options.folder+'_splits/')
    config.RESULT_DIR = os.path.join(config.PACKAGE_DIR, '../explanatoin_data/'+options.folder+'_explanations/')
    config.LABEL_FREQUENCY_THRESHOLD = options.threshold
    config.NEIGHBOURHOOD_SIZE = options.ksize
    
    config.RULE_METRICS = ['coverage', 'accuracy']
    if options.algorithm not in config.RULE_METRICS:
        config.RULE_METRICS.append(options.algorithm)
     
#     config.SAMPLING = options.sampling / 100.0
    config.NO_DISLIKES = True
    config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = options.discountThreshold
    config.EXPL_SAMPLE_SIZE = 1000
    
    if not os.path.exists(config.SPLIT_DIR):
        os.makedirs(config.SPLIT_DIR)
    if not os.path.exists(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)
    
    
    filenames = dataPreprocessing.loadData(mode='explanations',mean_center=options.meanCenter)
    # 5-fold cross-validation
    for iteration, (train_filename, test_filename, user_means_filename, eval_item_filename, opinion_filename) in enumerate(filenames, 1):
        
        config.CONT_SERENDIPITIES.clear()
        
        mean_centered = ''
        if options.meanCenter:
            mean_centered = 'meanCentered_'
        
        extended_candidates = ''
        if options.extendedCandidates:
            extended_candidates = 'extendedCandidates_'
        
#         sampled=''
#         if options.sampling:
#             sampled='samp'+str(options.sampling)+'_'
        
        acc_filtered=''
        if options.accuracyFilter:
            acc_filtered='accfiltered_'
        
        filtered_by_better_expl=''
        if options.discountThreshold:
            filtered_by_better_expl='filtered_by_better_expl_'
        
        # create the training data
        train_data = trainData.TrainData(train_filename, user_means_filename)
        
        # load the item opinion index
        config.ITEM_OPINIONS.clear()
        with open(opinion_filename, 'rb') as opinion_file:
            config.ITEM_OPINIONS = pickle.load(opinion_file)
        
        # run the beyondAccuracy for all 5-star and 1-star items
        logging.info('running explanations with {0}, {1}...'.format(test_filename, opinion_filename))
        
#         bad_items = []
#         good_items = []
        
        good_items_result_object = explanationResults.ExplanationResults()
        good_items_result_object.file = open(config.RESULT_DIR+options.dataset+'_good_explanations_'+options.algorithm+'_'+mean_centered+filtered_by_better_expl+acc_filtered+extended_candidates+str(iteration)+'.dat','w')
        
        with open(eval_item_filename,'rb') as expleval_file:
            
            for line in expleval_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1].rstrip('\n')
                
                # need to skip users and items who don't appear in training data
                assert user_id in train_data._row_indices
                assert item_id in train_data._col_indices
                
                user_index = train_data.getUserIndex(user_id)
                
                # THIS IS A FIX FOR THE FEW USERS IN LAST.FM WHO HAVE IDENTICAL RATINGS FOR ALL TRAIN ITEMS
                # WE HAVE TO SKIP THOSE USERS, BECAUSE THEY LOOSE ALL RATINGS IN THE MEAN-CENTERED MATRIX!
                if len(train_data.getUserProfileByIndex(user_index)) < 1:
                    continue
                
# #                 if rating == 1.0:
# #                     bad_items.append((user_id, item_id))
#                 if rating == 5.0:
#                     good_items.append((user_id, item_id))
        
#         logging.info('found {0} 5-star and {1} 1-star items'.format(len(good_items), len(bad_items)))
#         logging.info('found {0} 5-star items'.format(len(good_items)))
                
                logging.info('\t doing user {0} item {1}'.format(user_id, item_id))
                raw_string = good_items_result_object.computePerformanceMetrics(user_index, item_id, train_data, options)
                good_items_result_object.file.write(raw_string)
                good_items_result_object.file.flush()
        
        good_items_result_object.averageMetricValues(config.EXPL_SAMPLE_SIZE)
        good_items_result_object.file.close()
        
        
#         bad_items_result_object = explanationResults.ExplanationResults()
#         bad_items_result_object.file = open(config.RESULT_DIR+options.dataset+'_bad_explanations_'+options.algorithm+'_'+mean_centered+filtered_by_better_expl+acc_filtered+extended_candidates+str(iteration)+'.dat','w')
#         
#         for user_id, item_id in random.sample(bad_items, options.sample):
#             logging.info('\t doing user {0} item {1}'.format(user_id, item_id))
#             user_index = train_data.getUserIndex(user_id)
#             raw_string = bad_items_result_object.computePerformanceMetrics(user_index, item_id, train_data, options)
#             bad_items_result_object.file.write(raw_string)
#             bad_items_result_object.file.flush()
#         
#         bad_items_result_object.averageMetricValues(options.sample)
#         bad_items_result_object.file.close()
        
        
#         sys.stdout.write('iteration #'+str(iteration)+' completed. results:\n'+\
#                          bad_items_result_object.result_string+'\n------------------------\n'+good_items_result_object.result_string+'\n------------------------\n')
        sys.stdout.write('iteration #'+str(iteration)+' completed. results:\n'+\
                         good_items_result_object.result_string+'\n------------------------\n')
        sys.stdout.flush()
        
        
        