'''
Created on 13 Jan 2015

a module for data pre-processing operations: splitting rating files for cross-validation, preparing test item files 

@author: mkaminskas
'''

import glob
import logging
import os
import pickle
import random

from mrec.evaluation.preprocessing import TSVParser, SplitCreator
from mrec.examples.filename_conventions import get_splitfile
from mrec.examples.prepare import Processor

import dataReading
import numpy as np
from utils import config


def _splitData(data_tuples):
    '''
    create the CV splits from movie/music rating data
    uses the mrec library script
    
    @param data_tuples: the list of (user, item, rating) tuples
    @type data_tuples: list of tuples
    '''
    
    tmp_file_path = config.SPLIT_DIR+config.MOVIES_OR_MUSIC
    # write tmp file with tab-separated tuples (userID + offset, itemID + offset, rating)
    # this is needed because SplitCreator subtracts index offset when creating splits,
    # while we need them to contain item ids WITHOUT the offset
    with open(tmp_file_path,'wb') as tmp_file:
        
        data_tuples = sorted(data_tuples, key=lambda d: (d[0], d[1], d[2]))
        for data_tup in data_tuples:
            tmp_file.write(str(data_tup[0])+'\t'+str(data_tup[1])+'\t'+ str(data_tup[2])+'\n')
#             tmp_file.write(str(data_tup[0]+config.MREC_INDEX_OFFSET)+'\t'+str(data_tup[1]+config.MREC_INDEX_OFFSET)+'\t'+ str(data_tup[2])+'\n')
    
    data_path = os.path.abspath(tmp_file_path)
    outdir_path = os.path.abspath(config.SPLIT_DIR)
    rating_thresh = 0
    binarize = False
    delimiter = '\t'
    test_size = config.TEST_SIZE
    normalize = False
    discard_zeros = False
    min_items_per_user = 10
    num_splits = 5
    
    parser = TSVParser(thresh=rating_thresh, binarize=binarize, delimiter=delimiter)
    splitter = SplitCreator(test_size=test_size, normalize=normalize, discard_zeros=discard_zeros, sample_before_thresholding=False)
    processor = Processor(splitter, parser, min_items_per_user)
    
    filenames = []
    for i in xrange(num_splits):
        trainfile = get_splitfile(data_path, outdir_path, 'train', i)
        testfile = get_splitfile(data_path, outdir_path, 'test', i)
        
        logging.info('creating split {0}: {1} {2}'.format(i,trainfile,testfile))
        filenames.append((trainfile,testfile))
        processor.create_split(open(data_path),open(trainfile,'w'),open(testfile,'w'))
        
        too_few_items = processor.get_too_few_items()
        if (too_few_items):
            logging.info('skipped {0} users with less than {1} ratings'.format(too_few_items, min_items_per_user))
    logging.info('done!')
    
    return filenames


def _MeanCenter(data_file_path):
    '''
    user-mean-center rating data in the data file
    create a file with user means (needed for rating prediction)
    
    @param data_file_path: path to the data file to be mean-centered
    @type data_file_path: string
    
    @return: a tuple of normalized_file_path, user_means_filen_path
    @rtype: tuple
    '''
    
    centered_file_path = data_file_path+'.centered'
    user_means_file_path = data_file_path+'.means'
    
    logging.info('mean-centering into files {0}; {1}'.format(centered_file_path, user_means_file_path))
    
    previous_user = ''
    user_ratings = []
    user_items = []
    
    with open(data_file_path,'rb') as data_file:
        with open(centered_file_path,'wb') as centered_file:
            with open(user_means_file_path,'wb') as user_means_file:
                
                for line in data_file:
                    data = line.split('\t')
                    
                    user_id = str(data[0])
                    item_id = str(data[1])
                    rating = float(data[2])
                    
                    if user_id != previous_user:
                        
                        if user_ratings:
                            
                            user_mean = np.mean(user_ratings)
                            user_means_file.write(previous_user+'\t'+str(user_mean)+'\n')
                            
                            user_ratings = [r - user_mean for r in user_ratings]
                            
                            for i,r in zip(user_items, user_ratings):
                                centered_file.write(previous_user+'\t'+ i +'\t'+ str(r) +'\n')
                            
                        user_ratings = []
                        user_items = []
                    
                    previous_user = user_id
                    user_items.append(item_id)
                    user_ratings.append(float(rating))
                
                # and last time for the last user
                user_mean = np.mean(user_ratings)
                user_means_file.write(previous_user+'\t'+str(user_mean)+'\n')
                
                user_ratings = [r - user_mean for r in user_ratings]
                for i,r in zip(user_items, user_ratings):
                    centered_file.write(previous_user+'\t'+ i +'\t'+ str(r) +'\n')
    
    logging.info('done!')
    
    return centered_file_path, user_means_file_path


def loadData(mode, mean_center=False, one_test_item_per_user=True):
    '''
    the main data loading method
    
    mode = beyond_accuracy | explanations | survey
    
    if LOAD_DATA is True:
    - reads the raw rating data from files
    - splits data for CV
    - normalizes the data
    
    if LOAD_OPINIONS is True:
    - reads opinions from the rating files
    - stores them in opinion files (one per split)
    
    if LOAD_DATA is False:
    - reads filenames from the split directory
    
    if LOAD_OPINIONS is False:
    - reads filenames from the split directory
    
    also loads the item content labels into the global variables
    
    @return: list of (train_filename, test_filename, user_means_filename, opinion_filename) tuples
    @rtype list
    '''
    
    data_filenames = []
    
    # load item IDs, labels and the user IDs into global variables
    dataReading._readItemData()
    dataReading._readUserIds()
    
    
    if (config.LOAD_DATA is None) or (config.LOAD_OPINIONS is None):
        raise ValueError('LOAD_DATA or LOAD_OPINIONS not set. Cannot continue.')
    
    # if data loading flag is set to True, read the rating data from file and split it for 5-fold CV
    elif config.LOAD_DATA:
        
        data_tuples = dataReading._readRatingData()
        
        split_filenames = _splitData(data_tuples)
        
        # that's in case we only need to run the beyondAccuracy file construction part
#         split_filenames = [('../MOV20m_splits/movies_new.train.0', '../MOV20m_splits/movies_new.test.0'),\
#                            ('../MOV20m_splits/movies_new.train.1', '../MOV20m_splits/movies_new.test.1'),\
#                            ('../MOV20m_splits/movies_new.train.2', '../MOV20m_splits/movies_new.test.2'),\
#                            ('../MOV20m_splits/movies_new.train.3', '../MOV20m_splits/movies_new.test.3'),\
#                            ('../MOV20m_splits/movies_new.train.4', '../MOV20m_splits/movies_new.test.4')]
#         split_filenames = [('/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.train.0', '/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.test.0'),\
#                            ('/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.train.1', '/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.test.1'),\
#                            ('/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.train.2', '/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.test.2'),\
#                            ('/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.train.3', '/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.test.3'),\
#                            ('/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.train.4', '/home/mkaminskas/thebeyond/MOV20m_splits/movies_new.test.4')]
        
        for trainfile, testfile in split_filenames:
            centered_filename, user_means_filename = _MeanCenter(trainfile)
            # create the evaluation item file
            if mode == 'explanations':
                eval_item_filename = loadExplanationEvalFile(centered_filename, testfile)
            elif mode == 'beyond_accuracy':
                eval_item_filename = loadEvaluationItemFile(centered_filename, testfile, one_test_item_per_user)
            else:
                eval_item_filename = None
            
            opinion_filename = loadItemOpinions(centered_filename, mean_center)
            
            data_filenames.append((centered_filename, testfile, user_means_filename, eval_item_filename, opinion_filename))
        
    # if data loading flag set to False, read filenames of existing splits
    elif not config.LOAD_DATA:
        
        centered_filenames = sorted(glob.glob(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.train.*.centered'))
        test_filenames = sorted(glob.glob(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.test.*'))
        user_means_filenames = sorted(glob.glob(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.train.*.means'))
        if mode == 'explanations':
            eval_item_filenames = sorted(glob.glob(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.train.*.expleval'))
        elif mode == 'beyond_accuracy':
            eval_item_filenames = sorted(glob.glob(config.SPLIT_DIR+config.MOVIES_OR_MUSIC+'.train.*.eval'))
        else:
            eval_item_filenames = [None, None, None, None, None]
        
        for centered_filename, testfile, user_means_filename, eval_item_filename in zip(centered_filenames, test_filenames, user_means_filenames, eval_item_filenames):
            opinion_filename = loadItemOpinions(centered_filename, mean_center)
            data_filenames.append((centered_filename, testfile, user_means_filename, eval_item_filename, opinion_filename))
    
    logging.info('loaded data files: {0}'.format(data_filenames))
    
    return data_filenames
    

def loadEvaluationItemFile(centered_train_filename, test_filename, one_test_item_per_user):
    '''
    if LOAD_DATA is True:
    - for each user in the split test file (if that user has enough 5-star items) write the beyondAccuracy items to a file
    if LOAD_DATA is False:
    - return the name of corresponding beyondAccuracy item file
    '''
    
    eval_item_filename = centered_train_filename.rstrip('.centered')+'.eval'
    
    if config.LOAD_DATA is None:
        raise ValueError('LOAD_DATA not set. Cannot continue.')
    
    elif config.LOAD_DATA:
    
        logging.info('creating test item file {0}'.format(eval_item_filename))
        
        users_train_items = {}
        users_test_items = {}
        users_five_star_items = {}
        
        with open(centered_train_filename,'rb') as centered_train_file:
            for line in centered_train_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                users_train_items.setdefault(user_id, []).append(item_id)
        
        all_train_items = set([item for sublist in users_train_items.values() for item in sublist])
        
        with open(test_filename,'rb') as test_file:
            for line in test_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                rating = float(data[2].rstrip('\n'))
                users_test_items.setdefault(user_id, []).append(item_id)
                # record items that received 5 stars AND are present in the training data
                if rating >= config.RATING_THRESHOLD and item_id in all_train_items:
                    users_five_star_items.setdefault(user_id, []).append(item_id)
        
        with open(eval_item_filename,'wb') as eval_item_file:
            
            # that's in case we only need to get 100 random users for the eval file
#             random_sample = random.sample(users_five_star_items.keys(), 1000)
#             for user_id in random_sample:
#                 five_star_items = users_five_star_items[user_id]
            
            for user_id, five_star_items in users_five_star_items.items():
                
                if one_test_item_per_user:
                    
                    np.random.shuffle(five_star_items)
                    ground_truth_items = five_star_items[:config.HIGHLY_RATED_ITEMS_NUM]
                    
                    rated_items = set(users_train_items[user_id]) | set(users_test_items[user_id])
                    unrated_items = list(all_train_items - rated_items)
                    np.random.shuffle(unrated_items)
                    random_unrated_items = unrated_items[:1000]
                    
                    eval_item_file.write(user_id+'\t'+','.join(ground_truth_items)+'\t'+','.join(random_unrated_items)+'\n')
                    
                else:
                    # fixing the '1 + random' methodology
                    # some code repetition, but who cares
                    rated_items = set(users_train_items[user_id]) | set(users_test_items[user_id])
                    unrated_items = list(all_train_items - rated_items)
                    for five_star_item in five_star_items:
                        np.random.shuffle(unrated_items)
                        random_unrated_items = unrated_items[:1000]
                        eval_item_file.write(user_id+'\t'+five_star_item+'\t'+','.join(random_unrated_items)+'\n')
                        
        
        logging.info('done!')
        
    elif not config.LOAD_DATA:
        pass
    
    return eval_item_filename



def loadExplanationEvalFile(centered_train_filename, test_filename, one_test_item_per_user=True):
    '''
    if LOAD_DATA is True:
    - for each user in the split test file (if that user has a 5-star item) write the user and one random 5-star item to a file
    if LOAD_DATA is False:
    - return the name of corresponding beyondAccuracy item file
    '''
    
    expl_eval_filename = centered_train_filename.rstrip('.centered')+'.expleval'
    
    if config.LOAD_DATA is None:
        raise ValueError('LOAD_DATA not set. Cannot continue.')
    
    elif config.LOAD_DATA:
        
        logging.info('creating explanation beyondAccuracy file {0}'.format(expl_eval_filename))
        
        users_train_items = {}
        users_five_star_items = {}
        
        with open(centered_train_filename,'rb') as centered_train_file:
            for line in centered_train_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                users_train_items.setdefault(user_id, []).append(item_id)
        
        all_train_items = set([item for sublist in users_train_items.values() for item in sublist])
        
        with open(test_filename,'rb') as test_file:
            for line in test_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                rating = float(data[2].rstrip('\n'))
                # record items that received 5 stars AND are present in the training data
                if rating >= config.RATING_THRESHOLD and item_id in all_train_items:
                    users_five_star_items.setdefault(user_id, []).append(item_id)
        
        with open(expl_eval_filename,'wb') as expl_eval_file:
            
            if one_test_item_per_user:
                for user_id, five_star_items in users_five_star_items.items():
                    np.random.shuffle(five_star_items)
                    ground_truth_items = five_star_items[:config.HIGHLY_RATED_ITEMS_NUM]
                    expl_eval_file.write(user_id+'\t'+','.join(ground_truth_items)+'\n')
                
            else:
                # only get K random users for the eval file
                random_sample = random.sample(users_five_star_items.keys(), config.EXPL_SAMPLE_SIZE)
                for user_id in random_sample:
                    five_star_items = users_five_star_items[user_id]
                    np.random.shuffle(five_star_items)
                    expl_eval_file.write(user_id+'\t'+five_star_items[0]+'\n')
        
        logging.info('done!')
        
    elif not config.LOAD_DATA:
        pass
    
    return expl_eval_filename


def loadItemOpinions(centered_train_filename, mean_center):
    '''
    if LOAD_OPINIONS is True:
    - reads the rating data from CV split files
    - creates a dictionary of {(item_index,opinion):[user indices]}, where opinion = {dislike/neutral/like}
    - loads the dict into a file
    
    if LOAD_OPINIONS is False:
    - return the name of corresponding opinion item file
    '''
    
    # if the opinions are based on mean-centered ratings, need to read the centered file, otherwise, the original split file
    if mean_center:
        data_filename = centered_train_filename
        opinion_filename = centered_train_filename+'.opinions'
    else:
        data_filename = centered_train_filename.rstrip('.centered')
        opinion_filename = centered_train_filename.rstrip('.centered')+'.opinions'
    
    
    if config.LOAD_OPINIONS is None:
        raise ValueError('LOAD_OPINIONS not set. Cannot continue.')
        
    elif config.LOAD_OPINIONS:
        
        logging.info('creating opinion file {0} from data file {1}'.format(opinion_filename, data_filename))
        
        tmp_opinions = {}
        
        with open(data_filename,'rb') as train_file:
            for line in train_file:
                data = line.split('\t')
                user_id = data[0]
                item_id = data[1]
                
                if mean_center:
                    rating = float(data[2].rstrip('\n'))
                    
                    if rating < 0:
                        tmp_opinions.setdefault((item_id,'dislike'), []).append(user_id)
                    elif rating > 0:
                        tmp_opinions.setdefault((item_id,'like'), []).append(user_id)
                    # opinions for rating==0.0 should never be needed (these ratings would not appear in the rating matrix anyway)
                    
                else:
                    rating = round(float(data[2].rstrip('\n')), 1)
                    
                    if rating < 3.0:
                        tmp_opinions.setdefault((item_id,'dislike'), []).append(user_id)
                    elif rating == 3.0:
                        tmp_opinions.setdefault((item_id,'neutral'), []).append(user_id)
                    else:
                        tmp_opinions.setdefault((item_id,'like'), []).append(user_id)
             
        logging.info('done. {0} items stored.'.format(len(tmp_opinions)))
        
        with open(opinion_filename, 'wb') as opinion_file:
            pickle.dump(tmp_opinions, opinion_file)
        
        logging.info('dict pickled to {0}'.format(opinion_filename))
        
    elif not config.LOAD_OPINIONS:
        pass
    
    return opinion_filename




