'''
Created on Dec 4, 2013

the module containing tests

@author: Marius
'''

import ast
import itertools
import logging
import operator
import optparse
import os
import pickle
import random
import sys
from time import time

from mrec import load_fast_sparse_matrix
from mrec.item_similarity.knn import CosineKNNRecommender
from mrec.mf.warp import WARPMFRecommender
from scipy import spatial
from sparsesvd import sparsesvd

from beyondAccuracy import reranking
from dataHandling import dataPreprocessing, dataReading
from dataModel import trainData
from explanations import explanationResults
from frameworkMetrics import topNLists, diversity, serendipity, novelty, \
    explanationMetrics
import numpy as np
from utils import config, scoreNorm


def testToyExample():
    '''
    TEST data
    Users: Jack, u2, u3, Zak, Me
    Movies: A, B, C, D, E
    
            A    B    C    D    E
    Jack    5    1    3    4    3
    u2           4    1
    u3           2              5
    Zak     4    1    4    5    4
    Me      5         4         3
    '''
    centered_file_path, user_means_file_path = dataPreprocessing._MeanCenter('../splits/TEST')
    
    data_m = trainData.TrainData(centered_file_path, user_means_file_path)
    
    # test the rating matrix
    assert data_m.rating_matrix.shape == (5,5)
    assert data_m.getItemNeighboursByIndex(data_m.getItemIndex('C'), None)[0][0] == data_m.getItemIndex('D')
    assert data_m.getNumOfItemRatersByIndex(data_m.getItemIndex('A')) == 3
    assert all(data_m.getItemProfileByIndex(data_m.getItemIndex('D')) == [data_m.getUserIndex('Jack'),data_m.getUserIndex('Zak')])
    assert data_m.getPopularityInfo()[0][0] == 'D'
    
    assert data_m.getUserNeighboursByIndex(data_m.getUserIndex('Jack'), None)[0][0] == data_m.getUserIndex('Zak')
    assert all(data_m.getUserProfileByIndex(data_m.getUserIndex('u2')) == [data_m.getItemIndex('B'),data_m.getItemIndex('C')])
    
    
    # test the user-user matrix
    jack = [5.0, 1.0, 3.0, 4.0, 3.0]
    mean_jack = np.mean([i for i in jack if i > 0.0])
    
    zak = [4.0, 1.0, 4.0, 5.0, 4.0]
    mean_zak = np.mean([i for i in zak if i > 0.0])
    
    me = [5.0, 0.0, 4.0, 0.0, 3.0]
    mean_me = np.mean([i for i in me if i > 0.0])
    
    for i in range(len(jack)):
        if jack[i] > 0.0:
            jack[i] -= mean_jack
    for i in range(len(zak)):
        if zak[i] > 0.0:
            zak[i] -= mean_zak
    for i in range(len(me)):
        if me[i] > 0.0:
            me[i] -= mean_me
    
    my_sim = data_m.user_similarity_matrix[data_m.getUserIndex('Me'), data_m.getUserIndex('Jack')] 
    ground_truth_sim = 1 - spatial.distance.cosine(me, jack)
    assert abs(ground_truth_sim - my_sim) < 0.001
    
    
    # test the item-item matrix
    a = [5.0-mean_jack, 0.0, 0.0, 4.0-mean_zak, 5.0-mean_me]
    d = [4.0-mean_jack, 0.0, 0.0, 5.0-mean_zak, 0.0]
    
    my_sim = data_m.item_similarity_matrix[data_m.getItemIndex('A'), data_m.getItemIndex('D')] 
    ground_truth_sim = 1 - spatial.distance.cosine(a, d)
    assert abs(ground_truth_sim - my_sim) < 0.001
    
    
    # test recommendation generation
#     user_id = 'Me'
#     evaluation_item_ids = ['A', 'B', 'C', 'D', 'E']
#     
#     _, _, Q = sparsesvd(data_m.rating_matrix.tocsc(), 2)
#     mf = data_m.getFactorBasedRecommendations(user_id, Q, evaluation_item_ids)
#     
#     ub_classic = data_m.getUserBasedRecommendations(user_id, evaluation_item_ids, 'classic', verbose=True)
#     ub_damping = data_m.getUserBasedRecommendations(user_id, evaluation_item_ids, 'self_damping', verbose=True)
#     ub_non = data_m.getUserBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized', verbose=True)
#     
#     ib_classic = data_m.getItemBasedRecommendations(user_id, evaluation_item_ids, 'classic')
#     ib_damping = data_m.getItemBasedRecommendations(user_id, evaluation_item_ids, 'self_damping')
#     ib_non = data_m.getItemBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized')
#     
#     print mf
#     print '---------------------'
#     print ub_classic
#     print ub_damping
#     print ub_non
#     print '---------------------'
#     print ib_classic
#     print ib_damping
#     print ib_non
    
    # test diversity frameworkMetrics
    config.MOVIES_OR_MUSIC = 'movies'
    config.ITEM_DATA = {'A':{'labels':['horror']}, 'B':{'labels':['drama']}, 'C':{'labels':['drama']}, 'D':{'labels':['horror', 'comedy']}, 'E':{'labels':['drama']}}
    item_list = [('A',1.0),('B',1.0),('C',1.0),('D',0.5)]
    
    ground_truth_div = ((1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('A'), data_m.getItemIndex('B')]) / 2.0 + \
                        (1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('A'), data_m.getItemIndex('C')]) / 2.0 + \
                        (1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('A'), data_m.getItemIndex('D')]) / 2.0 + \
                        (1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('B'), data_m.getItemIndex('C')]) / 2.0 + \
                        (1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('B'), data_m.getItemIndex('D')]) / 2.0 + \
                        (1.0 - data_m.item_similarity_matrix[data_m.getItemIndex('C'), data_m.getItemIndex('D')]) / 2.0) / 6.0
    
    assert diversity.getListDiversity(data_m, item_list, 'div_r') == ground_truth_div
    assert diversity.getListDiversity(data_m, item_list, 'div_c') == 4.5 / 6
    
    # test serendipity frameworkMetrics
    assert serendipity.getListSerendipity(data_m, data_m.getUserIndex('Me'), [('A',0.5)], 'coocc') == 0.0
    assert serendipity.getListSerendipity(data_m, data_m.getUserIndex('Me'), [('D',0.5)], 'cont') == 0.5
    assert abs(serendipity.getListSerendipity(data_m, data_m.getUserIndex('Me'), item_list, 'coocc') - 0.306732842163) < 0.001
    
    # test novelty frameworkMetrics
    assert novelty._getItemNovelty(data_m, 'B') == novelty._getItemNovelty(data_m, 'E')
    
    # test re-ranking
    


def testDataSplitFiles():
    '''
    test if the files are split correctly:
    - splits are done for all users in the dataset;
    - no overlapping items in the user's split;
    - no items are lost on the split
    then test if the test item files are generated correctly
    '''
    
    logging.info('testing split files...')
    
    if not config.MOVIES_OR_MUSIC:
        raise ValueError('MOVIES_OR_MUSIC not set. Cannot continue.')
    elif config.MOVIES_OR_MUSIC == 'movies':
        data_file = config.MOVIE_RATINGS
    elif config.MOVIES_OR_MUSIC == 'music':
        data_file = config.ARTIST_RATINGS
    
    # a dict of {user : [rated items]}
    users_rated_items = {}
    
    with open(data_file,'rb') as data_file:
        for line in data_file:
            data = line.split('::')
            user_id = data[0]
#             item_id = config.ITEM_IDS[data[1]]
            item_id = data[1]
            users_rated_items.setdefault(user_id, []).append(item_id)
    
    for i in range(5):
        # dicts of users and their train/test items
        users_train_items = {}
        users_test_items = {}
        
        with open(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.train.'+str(i),'rb') as train_file:
            for line in train_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                users_train_items.setdefault(user_id, []).append(item_id)
        
        with open(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.test.'+str(i),'rb') as test_file:
            for line in test_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                users_test_items.setdefault(user_id, []).append(item_id)
        
        for userID in users_rated_items:
            
            if userID not in users_train_items:
                print userID#, [k for k,v in config.USER_IDS.items() if v == userID]
            else:
                assert (userID in users_train_items) and (userID in users_test_items)
                assert not any([itemID in users_train_items[userID] for itemID in users_test_items[userID]])
                assert set(users_train_items[userID] + users_test_items[userID]) == set(users_rated_items[userID])
    
    logging.info('done!')


def testDataCentering(train_filename, user_means_filename):
    '''
    test if the split file is mean-centered correctly:
    - user means are computed correctly
    - if the mean-centered ratings are stored correctly
    '''
    
    logging.info('testing data centering for files {0}; {1}...'.format(train_filename, user_means_filename))
    
    user_split_ratings = {}
    user_centered_ratings = {}
    user_means = {}
    
    original_split_filename = train_filename.rstrip('.centered')
    
    with open(original_split_filename,'rb') as split_file:
        for line in split_file:
            data = line.split('\t')
            user_id = data[0]
            rating = float(data[2])
            user_split_ratings.setdefault(user_id, []).append(rating)
    
    with open(train_filename,'rb') as centered_split_file:
        for line in centered_split_file:
            data = line.split('\t')
            user_id = data[0]
            rating = float(data[2])
            user_centered_ratings.setdefault(user_id, []).append(rating)
    
    with open(user_means_filename,'rb') as mean_file:
        for line in mean_file:
            data = line.split('\t')
            user_means[data[0]] = float(data[1])
    
    for userID in user_split_ratings:
        
        assert abs(user_means[userID] - np.mean(user_split_ratings[userID])) < 0.0001
        
        a = [round(r + user_means[userID]) for r in user_centered_ratings[userID]]
        b = user_split_ratings[userID]
        a.sort()
        b.sort()
        
        assert a == b
    
    logging.info('done!')
 
 
def testEvaluationItemFile(train_filename, test_filename, eval_item_filename):
    '''
    test if the test item file for a particular split is generated correctly:
    - the number of random items is correct;
    - no random items among the user's train/test ratings;
    - all ground truth items in uers' test ratings
    '''
    
    logging.info('testing test item file {0} for split {1}; {2}...'.format(eval_item_filename, train_filename, test_filename))
    
    users_train_items = {}
    users_test_items = {}
    users_5star_items = {}
    
    with open(train_filename,'rb') as train_file:
        for line in train_file:
            data = line.split('\t')
            user_id = data[0]
            item_id = data[1]
            users_train_items.setdefault(user_id, []).append(item_id)
    
    with open(test_filename,'rb') as test_file:
        for line in test_file:
            data = line.split('\t')
            user_id = data[0]
            item_id = data[1]
            rating = float(data[2].rstrip('\n'))
            users_test_items.setdefault(user_id, []).append(item_id)
            if rating >= config.RATING_THRESHOLD:
                users_5star_items.setdefault(user_id, []).append(item_id)
    
    with open(eval_item_filename,'rb') as eval_file:
        for line in eval_file:
            data = line.split('\t')
            user_id = data[0]
            gt_item_id = data[1]
            random_unrated_items = data[2].rstrip('\n').split(',')
            
            assert len(random_unrated_items) == 1000
            assert not any(i in users_train_items[user_id] for i in random_unrated_items)
            assert not any(i in users_test_items[user_id] for i in random_unrated_items)
            assert gt_item_id in users_test_items[user_id]
            assert gt_item_id in users_5star_items[user_id]
            assert gt_item_id not in users_train_items[user_id]
    
    logging.info('done!')


def testPredictionMethods(train_filename, eval_item_filename, user_means_filename):
    '''
    compare predictions generated by the different approaches
    computes pairwise list overlap and average recall for each method
    '''
    
    logging.info('testing predictions with data files {0}; {1}; {2}...'.format(train_filename, eval_item_filename, user_means_filename))
    
    
    mrec_train_data = load_fast_sparse_matrix('tsv', train_filename)
    
    mrec_recommender = CosineKNNRecommender(config.NEIGHBOURHOOD_SIZE)
    mrec_recommender.fit(mrec_train_data)
    
    warp_recommender = WARPMFRecommender(d=50, gamma=0.01, C=100.0)
    warp_recommender.fit(mrec_train_data.X)
    
    train_data = trainData.TrainData(train_filename, user_means_filename)
    _, _, Q = sparsesvd(train_data.rating_matrix.tocsc(), config.FACTOR_MODEL_SIZE)
    
    recalls = {}
    overlaps = {}
    top_recs = {}
    user_counter = 0.0
    methods = ['mrec', 'warp', 'mf', 'ub_classic', 'ib_classic', 'ub_damping', 'ib_damping', 'ub_non', 'ib_non']
    
    with open(eval_item_filename,'rb') as eval_file:
        for line in eval_file:
            data = line.split('\t')
            user_id = data[0]
            ground_truth_items = data[1].split(',')
            random_unrated_items = data[2].rstrip('\n').split(',')
            
            evaluation_item_ids = ground_truth_items + random_unrated_items
            
            # for each prediction method, compute topN recommendations once per user
            predictions1 = mrec_recommender.recommend_items(mrec_train_data.X, int(user_id)-config.MREC_INDEX_OFFSET, max_items=10000, return_scores=True)
            top_recs['mrec'] = topNLists.getTopNList(predictions1, evaluation_item_ids=evaluation_item_ids)
            
            predictions2 = warp_recommender.recommend_items(mrec_train_data.X, int(user_id)-config.MREC_INDEX_OFFSET, max_items=10000, return_scores=True)
            top_recs['warp'] = topNLists.getTopNList(predictions2, evaluation_item_ids=evaluation_item_ids)
            
            predictions3 = train_data.getFactorBasedRecommendations(user_id, Q, evaluation_item_ids)
            top_recs['mf'] = topNLists.getTopNList(predictions3)
            
            predictions4 = train_data.getUserBasedRecommendations(user_id, evaluation_item_ids, 'classic')
            top_recs['ub_classic'] = topNLists.getTopNList(predictions4)
            
            predictions5 = train_data.getItemBasedRecommendations(user_id, evaluation_item_ids, 'classic')
            top_recs['ib_classic'] = topNLists.getTopNList(predictions5)
            
            predictions6 = train_data.getUserBasedRecommendations(user_id, evaluation_item_ids, 'self_damping')
            top_recs['ub_damping'] = topNLists.getTopNList(predictions6)
            
            predictions7 = train_data.getItemBasedRecommendations(user_id, evaluation_item_ids, 'self_damping')
            top_recs['ib_damping'] = topNLists.getTopNList(predictions7)
            
            predictions8 = train_data.getUserBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized')
            top_recs['ub_non'] = topNLists.getTopNList(predictions8)
            
            predictions9 = train_data.getItemBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized')
            top_recs['ib_non'] = topNLists.getTopNList(predictions9)
            
            # then, use the computed topN lists to update recall and overlap values
            for method1 in methods:
                if method1 in recalls:
                    recalls[method1] += topNLists.getRecall(ground_truth_items, top_recs[method1])
                else:
                    recalls[method1] = topNLists.getRecall(ground_truth_items, top_recs[method1])
                
                for method2 in methods:
                    dict_key = method1 + '_' + method2
                    if dict_key in overlaps:
                        overlaps[dict_key] += topNLists.computeRecommendationListOverlap(top_recs[method1], top_recs[method2])
                    else:
                        overlaps[dict_key] = topNLists.computeRecommendationListOverlap(top_recs[method1], top_recs[method2])
            
            user_counter += 1.0
            logging.info('Tested user {0}. Current recalls: {1}. Current overlaps: {2}'.\
                         format(user_id, [(k, v/user_counter) for k,v in recalls.items()], [(k, v/user_counter) for k,v in overlaps.items()]))
            
    return recalls, overlaps


def testBeyondAccurracyMetrics(train_filename, eval_item_filename, user_means_filename):
    
    logging.info('testing beyond-accuracy topNLists with data files {0}; {1}; {2}...'.format(train_filename, eval_item_filename, user_means_filename))
    
    train_data = trainData.TrainData(train_filename, user_means_filename)
    _, _, Q = sparsesvd(train_data.rating_matrix.tocsc(), config.FACTOR_MODEL_SIZE)
    
    with open(eval_item_filename,'rb') as eval_file:
        for line in eval_file:
            data = line.split('\t')
            user_id = data[0]
            user_index = train_data.getUserIndex(user_id)
            
            if len(train_data.getUserProfileByIndex(user_index)) < 1:
                continue
            
            ground_truth_items = data[1].split(',')
            random_unrated_items = data[2].rstrip('\n').split(',')
             
            evaluation_item_ids = ground_truth_items + random_unrated_items
             
            rec_list_szie = config.RECOMMENDATION_LIST_SIZE * config.DIVERSIFICATION_CANDIDATES_FACTOR
            
#             predictions = train_data.getFactorBasedRecommendations(user_id, Q, evaluation_item_ids)
#             top_recs = topNLists.getTopNList(predictions, rec_list_szie)
            
#             predictions_ib = train_data.getItemBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized')
#             top_recs_ib = topNLists.getTopNList(predictions_ib, rec_list_szie)
            
#             predictions = library_recommender.recommend_items(mrec_train_data.X, int(user_id)-config.MREC_INDEX_OFFSET, max_items=10000, return_scores=True)
#             top_recs = topNLists.getTopNList(predictions, rec_list_szie, evaluation_item_ids)
            
            predictions_ub = train_data.getUserBasedRecommendations(user_id, evaluation_item_ids, 'non_normalized')
            top_recs_ub = topNLists.getTopNList(predictions_ub, rec_list_szie)
            
#             print 'user',user_id
            
#             print top_recs_ib, top_recs_ub
            
#             rare = train_data.getPopularityInfo()[:10]
#             pop = train_data.getPopularityInfo()[-10:]
            
            top_recs = top_recs_ub
            print 'diversity_ratings',diversity.getListDiversity(train_data, top_recs, 'div_r')
            print 'diversity_content',diversity.getListDiversity(train_data, top_recs, 'div_c')
            print 'content',serendipity.getListSerendipity(train_data, user_index, top_recs, 'sur_c')
            
#             print 'rare cooccurrence',serendipity.getListSerendipity(train_data, user_index, rare, 'sur_r')
#             print 'rare cooccurrence normalized',serendipity.getListSerendipity(train_data, user_index, rare, 'sur_r_n')
#             
#             print 'pop cooccurrence',serendipity.getListSerendipity(train_data, user_index, pop, 'sur_r')
#             print 'pop cooccurrence normalized',serendipity.getListSerendipity(train_data, user_index, pop, 'sur_r_n')
#             
#             print 'rare novelty',novelty.getListNovelty(train_data, rare)
#             
#             print 'pop novelty',novelty.getListNovelty(train_data, pop)
            
            print '------------------------------'


def testItemContentLabels(train_filename, eval_item_filename, user_means_filename):
    
    logging.info('testing if all items have content labels with data files {0}; {1}; {2}...'.format(train_filename, eval_item_filename, user_means_filename))
    
    train_data = trainData.TrainData(train_filename, user_means_filename)
    _, _, Q = sparsesvd(train_data.rating_matrix.tocsc(), config.FACTOR_MODEL_SIZE)
    
    for item_index in train_data._col_indices.values():
        item_id = train_data.getItemId(item_index)
        
        
        if item_id not in config.ITEM_DATA:
            print 'index',item_index,'id',item_id
            print len(config.ITEM_DATA)
        
        
        assert item_id in config.ITEM_DATA
    
    logging.info('done! tested {0} items. Average num of content labels is {1}'.format( len(train_data._col_indices), np.mean([len(item_dict['labels']) for item_dict in config.ITEM_DATA.values()]) ))
    
    with open(eval_item_filename,'rb') as eval_file:
        for line in eval_file:
            data = line.split('\t')
            user_id = data[0]
            user_index = train_data.getUserIndex(user_id)
            
            if len(train_data.getUserProfileByIndex(user_index)) < 1:
                continue
            
            ground_truth_items = data[1].split(',')
            random_unrated_items = data[2].rstrip('\n').split(',')
             
            evaluation_item_ids = ground_truth_items + random_unrated_items
            rec_list_szie = config.RECOMMENDATION_LIST_SIZE * config.DIVERSIFICATION_CANDIDATES_FACTOR
            predictions = train_data.getFactorBasedRecommendations(user_id, Q, evaluation_item_ids)
            top_recs = topNLists.getTopNList(predictions, rec_list_szie)
            
            print 'diversity_content',diversity.getListDiversity(train_data, top_recs, 'div_c')
            
            exit()
    


def testExplanations(train_data, test_filename, mean_center, n_users, n_recs, verb):
    
    objective_metrics = [\
                        'accuracy',\
                        'accuracy_ex',\
                        'info_gain',\
                        'inverted_info_gain',\
                        'discounted_accuracy_thresh',\
                        'discounted_accuracy_thresh_y',\
                        ]
    
    logging.info('testing explanations with data files {0}; {1}. and metrics {2}...'.format(train_filename, test_filename, objective_metrics))
    
    
    # select n_users random user IDs to read from the file
    random_user_ids = random.sample(train_data._row_indices.keys(), n_users)
    
    # find n_recs 5* items for the N random users, store them as {user: [item1, item2, ...]}
    evaluation_dict = {}
    with open(test_filename,'rb') as test_file:
        
        for line in test_file:
            data = line.split('\t')
            user_id = data[0]
            item_id = data[1]
            rating = float(data[2].rstrip('\n'))
            
#             if (user_id not in ['5854']) or (rating != 5.0):#'4836 5658 1890
#                 continue
            if (user_id not in random_user_ids) or (rating != 5.0):
                continue
#             if item_id not in ['1387']:
#                 continue
            
            # need to skip users and items who don't appear in training data
            if (user_id not in train_data._row_indices) or (item_id not in train_data._col_indices):
                continue
            if (user_id in evaluation_dict) and (len(evaluation_dict[user_id]) >= n_recs):
                continue
            
            evaluation_dict.setdefault(user_id, []).append(item_id)
        
        logging.info('found {0} random users with 5-star items'.format(len(evaluation_dict)))
    
    
    for user_id, five_star_test_items in evaluation_dict.iteritems():
        
        user_index = train_data.getUserIndex(user_id)
        full_profile = train_data.getUserProfileByIndex(user_index)
         
#         if len(full_profile) > 100:
#             continue
        
        profile_string = ' '
        five_star_profile = train_data.getUserProfileByIndex(user_index, filter_by_rating=5.0)
        for i_ind in five_star_profile[:10]:
            
            profile_item_id = train_data.getItemId(i_ind)
            profile_string += 'ID-'+profile_item_id
            if config.ITEM_DATA[profile_item_id]['title']:
                profile_string += ' '+config.ITEM_DATA[profile_item_id]['title'].encode('utf8')+'; '
        
        print '\nuser ID-'+user_id+'\nprofile size: '+str(len(full_profile))+'; sample 5* items: '+profile_string.encode('utf8')
        
        for item_id in five_star_test_items:
            
            print '\ntest item:\t ID-'+item_id+' '+config.ITEM_DATA[item_id]['title'].encode('utf8'),'\t'
            
            novelties = {}
            
            for objective_metric in objective_metrics:
                t = time()
                
                # this is gonna get ugly, but for now I don't care
                # extracting parameters from metric names:
                ex_search = False
                if '_ex' in objective_metric:
                    ex_search = True
                    objective_metric = objective_metric.replace('_ex','')
                    
                acc_filter = False
                if '_y' in objective_metric:
                    acc_filter = True
                    objective_metric = objective_metric.replace('_y','')
                
                
                rule, rule_metrics = train_data.generateExplanations(user_index, item_id, objective_metric, extended_candidates=ex_search, acc_filter=acc_filter, verbose=verb)
                
                novelties[objective_metric] = novelty.getListNovelty(train_data, rule)
                
            ################### printing explanation ###############
                expl_string = '\n'+objective_metric+' (extended candidates='+str(ex_search)+')'+' explanation:\n'
                if rule:
                    rule_string = ''
                    for mov_id, mov_opinion in rule:
                        
                        mov_popularity = train_data.getNumOfItemRatersByIndex(train_data.getItemIndex(mov_id))
                        
                        if config.ITEM_DATA[mov_id]['title']:
                            rule_string += 'ID-'+mov_id+' ('+str(mov_popularity)+' raters) '+config.ITEM_DATA[mov_id]['title'].encode('utf8')+'--'+mov_opinion+';\t'
                        else:
                            rule_string += 'ID-'+mov_id+'--'+mov_opinion+';\t'
                    
                    expl_string += rule_string.rstrip(';\t')
                else:
                    expl_string += 'can\'t explain'
                
                print expl_string.encode('utf8')
                print 'and the rule metrics are',rule_metrics,'novelty:',novelties[objective_metric]
                 
                print(objective_metric+' explanations generated in %0.3fs.' % (time() - t))
            
            print '-----------------------------\n'
            
            ################### printing explanation ###############
            


def generateExplanationSurvey(train_data, test_filename, iteration, verb):
    
    objective_metrics = [\
                        'accuracy',\
                        'discounted_accuracy_thresh',\
                        'discounted_accuracy_thresh_better',\
                        ]
    
    if 'new' in config.MOVIES_OR_MUSIC:
        year_threshold = 2000
    else:
        year_threshold = 1996
    
    filtered_movie_ids = []
    
    for item_index in train_data._col_indices.values():
        item_id = train_data.getItemId(item_index)
        
        # check movie recency
        item_year = 0
        item_title = config.ITEM_DATA[item_id]['title']
        
        if '(' not in item_title:
            continue
        
        item_year_str = item_title.split('(')[1].rstrip(')')
        if item_year_str.isdigit():
            item_year = int(item_year_str)
        
        # check movie popularity
        num_of_item_ratings = train_data.getNumOfItemRatersByIndex(item_index)
        
        if (item_year >= year_threshold) and (num_of_item_ratings >= 100):
            filtered_movie_ids.append(item_id)
    
    logging.info('found {0} recent and popular movies.'.format(len(filtered_movie_ids)))
    
    
    filtered_explanation_candidates = []
    
    with open(test_filename,'rb') as test_file:
        
        for line in test_file:
            data = line.split('\t')
            user_id = data[0]
            item_id = data[1]
            rating = float(data[2].rstrip('\n'))
            
            if rating != 5.0:
                continue
            
            if item_id not in filtered_movie_ids:
                continue
            
            # check user profile size, if smaller than 20, skip
            u_index = train_data.getUserIndex(user_id)
            u_profile_indices = train_data.getUserProfileByIndex(u_index)
            if len(u_profile_indices) < 20:
                continue
            
            # skip items that are already present in the candidate pool
            if item_id in [t[1] for t in filtered_explanation_candidates]:
                continue
            
            filtered_explanation_candidates.append((u_index, item_id))
    
    logging.info('out of the recent and popular movies, {0} have 5-star ratings by prolific users.'.format(len(filtered_explanation_candidates)))
    
    
    sampled_explanation_candidates = random.sample(filtered_explanation_candidates, 100)
    test_case_counter = 0
    with open(os.path.join(config.PACKAGE_DIR, config.RESULT_DIR, 'survey'+str(iteration)+'.dat'),'a') as result_file:
    
        for u_idx, i_id in sampled_explanation_candidates:
            
            output_str = 'user id '+str(train_data.getUserId(u_idx))+' recommendation id '+str(i_id)+' '+config.ITEM_DATA[i_id]['title']+'\n'
            
            explanation_string = str(test_case_counter+1)+'.\tExplanations for the recommendation \"'+config.ITEM_DATA[i_id]['title']+'\":\n\t-----------------------------\n'
            
            rules = []
            random.shuffle(objective_metrics)
            for objective_metric in objective_metrics:
                
                if 'better' in objective_metric:
                    config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = 1
                    objective_metric = objective_metric.replace('_better','')
                else:
                    config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = 0
                
                rule, _ = train_data.generateExplanations(u_idx, i_id, objective_metric, extended_candidates=1, acc_filter=1, verbose=verb)
                if rule is None:
                    continue
                rules.append(rule)
                
                ################### explanation ###############
                zz=''
                if config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS:
                    zz = '(uniqueness)'
                explanation_string += '__\t'+objective_metric+zz+' explanation:\n'
                if rule:
                    rule_string = 'You liked '
                    for mov_id, _ in rule:
                        assert config.ITEM_DATA[mov_id]['title']
                        rule_string += '\"'+config.ITEM_DATA[mov_id]['title']+'\" and '
                        output_str += '\tmovie id '+str(mov_id)+' '+config.ITEM_DATA[mov_id]['title']+'\n'
                    
                    explanation_string += '\t'+rule_string.rstrip(' and ')+'.\n'
                    output_str += '-------\n'
                    
                else:
                    continue
                explanation_string += '\tPeople who like these movies also like \"'+config.ITEM_DATA[i_id]['title']+'\".\n\t-----------------------------\n'
                ################### end explanation ###############
            
            explanation_string += '__\tNone of the above explanations are helpful.\n\t-----------------------------\n'
            explanation_string += '__\tI don\'t know the movies well enough.'
            explanation_string += '\n\t-----------------------------\n\n\t-----------------------------\n'
            
            # making sure that all 3 methods generate a rule, all rules are longer than 1, and aren't overlapping 100%
            skip_it = False
            for rule1, rule2 in itertools.combinations(rules, 2):
                # if there is a rule exactly like one other, we need to skip it
                overlap = len(set(rule1) & set(rule2))
                if overlap == min(len(rule1), len(rule2)):
                    skip_it = True
            
            if (not skip_it) and (len(rules)==3) and (all(len(r)>1 for r in rules)):
                result_file.write(explanation_string)
                test_case_counter += 1
                result_file.flush()
                
                print 'a new test case created, currently have',test_case_counter
                print output_str
            else:
                print 'skipped a bad case'
            
            if test_case_counter == 20:
                break
    

if __name__ == "__main__":
    
#     testToyExample()
#     exit()
    
    ############ PARAMETER SETUP #############
    
    parser = optparse.OptionParser()
    parser.add_option("-i", "--iteration", type="int", action="store", dest="iteration", help="if >0, specifies the CV iteration to be used", default=0)
    (options, args) = parser.parse_args()
    
    config.LABEL_FREQUENCY_THRESHOLD = 10
    config.SPLIT_DIR = os.path.join(config.PACKAGE_DIR, '../explanation_data/MOVnew_splits/')# '../MOVlatest_splits/', '../MOV20m_splits/' , '../testing_explanations_splits/'
    config.RESULT_DIR = os.path.join(config.PACKAGE_DIR, '../explanation_data/MOVnew_explanations_results2/')
    config.MOVIES_OR_MUSIC = 'movies_new' # movies/new/newest
    config.LOAD_DATA = False
    config.LOAD_OPINIONS = False
    config.NEIGHBOURHOOD_SIZE = 150
    config.FACTOR_MODEL_SIZE = 25
    
    config.RULE_METRICS = ['coverage', 'accuracy', 'discounted_accuracy_thresh']
    config.NO_DISLIKES = True
    #config.DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = 1
    
    if not os.path.exists(config.SPLIT_DIR):
        os.makedirs(config.SPLIT_DIR)
    if not os.path.exists(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)
    
    filenames = dataPreprocessing.loadData(mode='survey')
#     testDataSplitFiles()
    
    for iteration, (train_filename, test_filename, user_means_filename, eval_item_filename, opinion_filename) in enumerate(filenames, 1):
        
        if options.iteration and (iteration != options.iteration):
            continue
        
        train_data = trainData.TrainData(train_filename, user_means_filename)
        
        config.ITEM_OPINIONS.clear()
        with open(opinion_filename, 'rb') as opinion_file:
            config.ITEM_OPINIONS = pickle.load(opinion_file)
        
#         testDataCentering(train_filename, user_means_filename)
#         testEvaluationItemFile(train_filename, test_filename, eval_item_filename)
#         recalls, overlaps = testPredictionMethods(train_filename, eval_item_filename, user_means_filename)
#         testItemContentLabels(train_filename, eval_item_filename, user_means_filename)
#         testBeyondAccurracyMetrics(train_filename, eval_item_filename, user_means_filename)
#         testExplanations(train_data, test_filename, mean_center=False, n_users=5, n_recs=5, verb=False)
#         explanationMetrics._getRuleDiscountedAccuracy(train_data, 0.58, [('22', 'like')], ('50', 'like'), set(config.ITEM_OPINIONS['22', 'like']), True)
#         explanationMetrics._getRuleDiscountedAccuracy(train_data, 0.46, [('1617', 'like')], ('50', 'like'), set(config.ITEM_OPINIONS['1617', 'like']), True)
        generateExplanationSurvey(train_data, test_filename, iteration, verb=False)
        
#         exit()
        