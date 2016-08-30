'''
Created on 18 May 2014

configuration module for data reading, pre-processing, and beyondAccuracy parameters

@author: mkaminskas
'''
import logging
import os

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(module)s.%(funcName)s(): %(message)s')

# do we work with movies or music?
# needs to be set from the main module (i.e., Tests or Evaluate)!
MOVIES_OR_MUSIC = None

# load the data or use the existing split/beyondAccuracy files?
# needs to be set from the main module (i.e., Tests or Evaluate)!
LOAD_DATA = None
LOAD_OPINIONS = None

# data files
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_RATINGS = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_1M/ratings.dat')
MOVIELENS_DATA = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_1M/movies.dat')
MOVIE_FILE_IMDB = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_1M/movies_imdb.dat')

MOVIE_RATINGS_NEW = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_10M/ratings.dat')
MOVIELENS_DATA_NEW = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_10M/movies.dat')

MOVIE_RATINGS_NEWEST = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_newest/ratings.csv')
MOVIELENS_DATA_NEWEST = os.path.join(PACKAGE_DIR, '../datasets/MovieLens_newest/movies.csv')

ARTIST_RATINGS = os.path.join(PACKAGE_DIR, '../datasets/LastFm_1K/ratings.dat')
LASTFM_DATA = os.path.join(PACKAGE_DIR, '../datasets/LastFm_1K/userid-timestamp-artid-artname-traid-traname.tsv')
ARTIST_FILE_LASTFM = os.path.join(PACKAGE_DIR, '../datasets/LastFm_1K/artists.dat')

SPLIT_DIR = ''
RESULT_DIR = ''

# data loading and pre-processing
TEST_SIZE = 0.2
MREC_INDEX_OFFSET = 1
MIN_ARTIST_POPULARITY = 20
LABEL_FREQUENCY_THRESHOLD = 1


# beyondAccuracy
NUM_USERS_MOVIELENS = None
NUM_USERS_LASTFM = None
NEIGHBOURHOOD_SIZE = None
FACTOR_MODEL_SIZE = None
RECOMMENDATION_LIST_SIZE = 10
DIVERSIFICATION_CANDIDATES_FACTOR = 5
RATING_THRESHOLD = 5.0
HIGHLY_RATED_ITEMS_NUM = 1
MAX_ITEM_COOCCURRENCE = None

# global dicts to store item IDs, titles, and labels as well as user IDs and rated items
# ITEM_DATA = {item_id : {'title':str, 'labels':[]}}
ITEM_DATA = {}
# USER_DATA = {user_id : [item_id1, item_id2, ...]}
USER_DATA = {}

# a global dict of pairwise item co-occurrence serendipities, for each item pair it only needs to be computed once
COOCC_SERENDIPITIES = {}
# a global dict of pairwise item content serendipities, for each item pair it only needs to be computed once 
CONT_SERENDIPITIES = {}
# a global dict for storing lists of users who dislike/are neutral/like an item (for faster explanation rule coverage lookup)
ITEM_OPINIONS = {} 
# a list of metrics to be computed for explanation rules
RULE_METRICS = []
# a flag for sampling items during explanation generation
# SAMPLING = None
# a flag for excluding dislikes/neutrals from explanations
NO_DISLIKES = None
# a flag for using the number of better explanations to discount accuracy
DISCOUNT_ACCURACY_BY_BETTER_EXPLANATIONS = None

EXPL_SAMPLE_SIZE = None
