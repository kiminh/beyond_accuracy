'''
Created on Sep 16, 2013

run beyondAccuracy of different recommendation approaches:
    - pureSVD as defined by [Cremonesi, Koren, Turin, 2009]
    - user-based CF
    - item-based CF
with various re-ranking techniques:
    - greedy re-ranking as defined by [McClave & Smyth, 2001]
    - greedy re-ranking using item novelty as the utility score

@author: Marius
'''

import logging
import optparse
import os
import pickle
import sys

from mrec import load_fast_sparse_matrix
from mrec.item_similarity.knn import CosineKNNRecommender
from mrec.mf.warp import WARPMFRecommender
from sparsesvd import sparsesvd

from beyondAccuracy import results
from dataModel import trainData
from frameworkMetrics import topNLists
from dataHandling import dataPreprocessing
from utils import config


def userEvaluation(result_object, train_data, Q, library_recommender, user_id, user_index, ground_truth_items, random_unrated_items, iteration, options):
    '''
    get performance frameworkMetrics for a given user,
    using the specified rec. algorithm and re-ranking approach
    
    update the relevant result objects and write raw data to a file 
    '''
    
    evaluation_item_ids = ground_truth_items + random_unrated_items
    rec_list_szie = config.RECOMMENDATION_LIST_SIZE * config.DIVERSIFICATION_CANDIDATES_FACTOR
    
    logging.info('\t doing user {0}'.format(user_id))
    
    # compute predictions
    if options.algorithm == 'mf':
        predictions = train_data.getFactorBasedRecommendations(user_id, Q, evaluation_item_ids)
        top_recs = topNLists.getTopNList(predictions, rec_list_szie)
        
    elif options.algorithm == 'ub':
        predictions = train_data.getUserBasedRecommendations(user_id, evaluation_item_ids, options.neighbours)
        top_recs = topNLists.getTopNList(predictions, rec_list_szie)
        
    elif options.algorithm == 'ib':
        predictions = train_data.getItemBasedRecommendations(user_id, evaluation_item_ids, options.neighbours)
        top_recs = topNLists.getTopNList(predictions, rec_list_szie)
        
    elif options.algorithm == 'mrec' or options.algorithm == 'warp':
        predictions = library_recommender.recommend_items(mrec_train_data.X, int(user_id)-config.MREC_INDEX_OFFSET, max_items=10000, return_scores=True)
        top_recs = topNLists.getTopNList(predictions, rec_list_szie, evaluation_item_ids)
        
    else:
        raise ValueError('Wrong rec. algorithm')
        
    raw_string = result_object.computePerformanceMetrics(user_index, top_recs, ground_truth_items, train_data, options)
    result_object.file.write(raw_string)
    result_object.file.flush()
    


if __name__ == "__main__":
    
    usage = "runEvaluation.py -d < movies | music > \
                         -a < mf | ub | ib | mrec | warp > \
                         -r < bs | div_c | div_r | sur_c | sur_r | sur_r_n | nov > \
                         -f < folder > \
                         -l < 0 | 1 > \
                         -n <'' | classic | non_normalized | self_damping> \
                         -t <1 - 10> \
                         -k <10 - 100> \
                         -s < 0 | 1 >"
    
    parser = optparse.OptionParser(usage)
    parser.add_option("-d", "--dataset", action="store", dest="dataset", help="the dataset to be used")
    parser.add_option("-a", "--algorithm", action="store", dest="algorithm", help="the recommendation algorithm")
    parser.add_option("-f", "--folder", action="store", dest="folder", help="the folder to store CV splits and beyondAccuracy")
    parser.add_option("-r", "--rerank", action="store", dest="rerank", help="the method to use for result re-ranking")
    parser.add_option("-l", "--loadData", type="int", action="store", dest="loadData", help="0/1 switch to generate splits or use existing ones", default=0)
    parser.add_option("-n", "--neighbours", action="store", dest="neighbours", help="the type of neighbourhood selection, leave empty algorithm is not ub or ib", default='')
    parser.add_option("-t", "--threshold", type="int", action="store", dest="threshold", help="the label frequency threshold for movie data", default=10)
    parser.add_option("-k", "--ksize", type="int", action="store", dest="ksize", help="the size of neighbourhood in knn approaches", default=50)
    parser.add_option("-s", "--sample", type="int", action="store", dest="sample", help="0/1 switch for user sampling to save time", default=0)
    parser.add_option("-c", "--nfactors", type="int", action="store", dest="nfactors", help="the number of factors in MF approach", default=25)
    parser.add_option("-g", "--cvalue", type="float", action="store", dest="cvalue", help="the C parameter in WARP approach", default=100.0)
    parser.add_option("-o", "--opinions", type="int", action="store", dest="opinions", help="0/1 switch for loading the item opinions", default=0)
    parser.add_option("-x", "--oneperuser", type="int", action="store", dest="oneperuser", help="0/1 switch for loading just one test item per user", default=1)
    (options, args) = parser.parse_args()
    
    if not (options.dataset and options.algorithm and options.rerank and options.folder):
        parser.error("the required options not entered")
    
    
    config.MOVIES_OR_MUSIC = options.dataset
    config.LOAD_DATA = options.loadData
    config.LOAD_OPINIONS = options.opinions
    config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, '../beyond_accuracy_data/'+options.folder+'_splits/')
    config.RESULT_DIR = os.path.join(config.PACKAGE_DIR, '../beyond_accuracy_data/'+options.folder+'_results/')
    config.LABEL_FREQUENCY_THRESHOLD = options.threshold
    config.NEIGHBOURHOOD_SIZE = options.ksize
    config.FACTOR_MODEL_SIZE = options.nfactors
    
    if options.sample:
        config.NUM_USERS_MOVIELENS = 1000
        config.NUM_USERS_LASTFM = 300
    else:
        config.NUM_USERS_MOVIELENS = 6040
        config.NUM_USERS_LASTFM = 1000
    
    
    if 'expl' in options.rerank:
        config.RULE_METRICS = ['coverage', 'accuracy']
        config.NO_DISLIKES = True
    
    
    if not os.path.exists(config.SPLIT_DIR):
        os.makedirs(config.SPLIT_DIR)
    if not os.path.exists(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)
        os.makedirs(config.RESULT_DIR+'coverage/')
    
    # hard-coded alpha values for the different re-ranking approaches
    div_c_alpha = 0.5
    div_r_alpha = 0.5
    sur_c_alpha = 0.5
    sur_r_alpha = 0.5
    sur_r_n_alpha = 0.5
    nov_alpha = 0.5
    
    
    filenames = dataPreprocessing.loadData(mode='beyond_accuracy',one_test_item_per_user=options.oneperuser)
    # 5-fold cross-validation
    for iteration, (train_filename, test_filename, user_means_filename, eval_item_filename, opinion_filename) in enumerate(filenames, 1):
        
#         if iteration in [1,2,3,4]:
#             continue
        
        # clear the global vars that store serendipity values for item pairs,
        # as they need to be re-populated for each fold
        config.COOCC_SERENDIPITIES.clear()
        config.CONT_SERENDIPITIES.clear()
        
        # create the result object and open the result file
        result_object = None
        result_file = None
        
        # a quick fix for file naming
        bubt=''
        if (options.algorithm == 'mf'):
            bubt = str(options.nfactors)
        elif (options.algorithm == 'warp'):
            bubt = str(options.nfactors)+'_'+str(options.cvalue)
        else:
            bubt = str(options.ksize)
        
        if options.rerank == 'bs':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_baseline_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, None, None)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_baseline.dat','a')
            
        elif options.rerank == 'div_c':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_diversity-content_'+str(div_c_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'div_c', div_c_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_diversity-content.dat','a')
            
        elif options.rerank == 'div_r':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_diversity-ratings_'+str(div_r_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'div_r', div_r_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_diversity-ratings.dat','a')
            
        elif options.rerank == 'sur_c':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-content_'+str(sur_c_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'sur_c', sur_c_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-content.dat','a')
            
        elif options.rerank == 'sur_r':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-ratings_'+str(sur_r_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'sur_r', sur_r_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-ratings.dat','a')
            
        elif options.rerank == 'sur_r_n':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-ratings-norm_'+str(sur_r_n_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'sur_r_n', sur_r_n_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_surprise-ratings-norm.dat','a')
            
        elif options.rerank == 'nov':
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_novelty_'+str(nov_alpha)+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, 'nov', nov_alpha)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_novelty.dat','a')
            
            
        # the new explanation re-ranking!
        elif 'expl' in options.rerank:
            
            result_file = open(config.RESULT_DIR+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_'+options.rerank+'_'+str(iteration)+'.dat','w')
            result_object = results.Results(options.algorithm, options.rerank, alpha=0.5)
            coverage_file = open(config.RESULT_DIR+'coverage/'+options.dataset+'_'+options.algorithm+options.neighbours+'_'+bubt+'_'+options.rerank+'.dat','a')
            
            # load the item opinion index
            config.ITEM_OPINIONS.clear()
            with open(opinion_filename, 'rb') as opinion_file:
                config.ITEM_OPINIONS = pickle.load(opinion_file)
            
            
        else:
            raise ValueError('Wrong reranking method entered. Choose between: bs | div_c | div_r | sur_c | sur_r | sur_r_n | nov')
            
        result_object.file = result_file
        
        # create the training data and required recommendation models
        train_data = trainData.TrainData(train_filename, user_means_filename)
        
        Q = None
        library_recommender = None
        
        if options.algorithm == 'mf':
            _, _, Q = sparsesvd(train_data.rating_matrix.tocsc(), config.FACTOR_MODEL_SIZE)
        elif options.algorithm == 'mrec':
            mrec_train_data = load_fast_sparse_matrix('tsv', train_filename)
            library_recommender = CosineKNNRecommender(config.NEIGHBOURHOOD_SIZE)
            library_recommender.fit(mrec_train_data)
        elif options.algorithm == 'warp':
            mrec_train_data = load_fast_sparse_matrix('tsv', train_filename)
            library_recommender = WARPMFRecommender(d=config.FACTOR_MODEL_SIZE, gamma=0.01, C=options.cvalue)
            library_recommender.fit(mrec_train_data.X)
        elif options.algorithm in ['ub', 'ib']:
            pass
        else:
            raise ValueError('Wrong rec. algorithm entered. Choose between ub, ib, mf, mrec, and warp')
        
        
        # run the beyondAccuracy for all users in the .eval file
        logging.info('running beyondAccuracy with {0}...'.format(eval_item_filename))
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
                
                userEvaluation(result_object, train_data, Q, library_recommender, user_id, user_index, ground_truth_items, random_unrated_items, iteration, options)
                # count the users for which predictions could be made (to use when averaging beyondAccuracy across users)
                evaluation_cases += 1
                
                # due to performance issues, limit the beyondAccuracy to N random users
                if options.dataset == 'movies' and evaluation_cases == config.NUM_USERS_MOVIELENS and options.oneperuser:
                    break
                elif options.dataset == 'music' and evaluation_cases == config.NUM_USERS_LASTFM and options.oneperuser:
                    break
        
        # average the beyondAccuracy by the number of test users, output them
        result_object.averageMetricValues(evaluation_cases)
        sys.stdout.write('iteration #'+str(iteration)+' completed. beyondAccuracy:\n'+result_object.result_string+'\n------------------------\n')
        
        # also write coverage data to a separate file
        coverage_file.write('iteration #'+str(iteration)+'\n'+str(result_object.coverage)+'\n')
        coverage_file.flush()
        
        sys.stdout.flush()
        
        # close the result file
        result_object.file.close()
        coverage_file.close()
        
    