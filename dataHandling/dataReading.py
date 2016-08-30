'''
Created on Oct 22, 2013

a module for data operations that involve reading from files

@author: Marius
'''

import csv
import io
import logging

from utils import config, counter


def _parseLastFMData():
    '''
    read the LastFM data (userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name)
    store as dictionary of {user_id : [artist_id1,artist_id2,artist_id1,...]}
    only stores the artists that are in ARTIST_FILE_LASTFM
    '''
    
    users_artists = {}
    
    identified_artists = set()
    # before reading the data, need to construct a list of already processed artists
    with open(config.ARTIST_FILE_LASTFM,'rb') as artists:
        for line in artists:
            data = line.split('::')
            identified_artists.add(data[0]) 
    
    logging.info('{0} identified artists found. Now processing the data file {1}...'.format(len(identified_artists), config.LASTFM_DATA))
    
    with open(config.LASTFM_DATA, 'rb') as f:
        for line in f:
            data = line.split('\t')
            user_id = data[0]
            artist_id = data[2]
            
            if artist_id in identified_artists:
                users_artists.setdefault(user_id, []).append(artist_id)
    
    artist_list = []
    for (user_id, artists) in users_artists.iteritems():
        #remove duplicates in user's artist list and add them to a single list for counting artist popularity
        d = {}.fromkeys(artists)
        artist_list += d.keys()
    artist_popularity = counter.Counter(artist_list)
    
    logging.info('done')
    
    return users_artists, artist_popularity



def _generateArtistRatings():
    '''
    convert artist access frequencies to [1..5] ratings as defined by [Vargas and Castells, 2011] and [Celma and Herrera, 2008]:
    r(u,i) ~ 5 * ( |{j s.t. freq(u,j) <= freq(u,i)}| / |u| ), where
    freq(u,i) denotes artist i access frequency by user u, |u| is the number of artists accessed by user u
    '''
    
    logging.info('generating artist ratings...')
    
    with open(config.ARTIST_RATINGS,'w') as f:
        users_artists, artist_popularity = _parseLastFMData()
        for user_id, artists in users_artists.iteritems():
            # count the frequencies of each artist the user u listened to
            artist_freqs = counter.Counter(artists)
            
            for artist_id in artist_freqs:
                #consider only artists that were listened to by at least X users
                if artist_popularity[artist_id] >= config.MIN_ARTIST_POPULARITY:
                    
                    # get the number of artists in artist_freqs that have frequency <= than the current artist_id
                    num = len([a_id for a_id, freq in artist_freqs.iteritems() if freq <= artist_freqs[artist_id]])
                    # make sure that the lowest rating is 1
                    rating = max(1, int( round(5.0 * float(num) / len(artist_freqs)) ))
                    
                    f.write(user_id+'::'+artist_id+'::'+str(rating)+'\n')
    
    logging.info('done')


def _readItemData():
    '''
    read the IDs and labels of a movie or an artist from a label file
    store them in the global variables
    
    possibly clean the labels to leave only those that appear in at least 10 items
    '''
    
    if not config.MOVIES_OR_MUSIC:
        raise ValueError('MOVIES_OR_MUSIC not set. Cannot continue.')
    elif config.MOVIES_OR_MUSIC == 'movies':
        logging.info('reading Movielens labels with a threshold of {0}...'.format(config.LABEL_FREQUENCY_THRESHOLD))
        label_path = config.MOVIE_FILE_IMDB
    elif config.MOVIES_OR_MUSIC == 'music':
        logging.info('reading Last.fm labels with a threshold of {0}...'.format(config.LABEL_FREQUENCY_THRESHOLD))
        label_path = config.ARTIST_FILE_LASTFM
    elif config.MOVIES_OR_MUSIC == 'movies_new':
        # the new datasets have no IMDB label file and only uses genres from the movies' file
        # therefore, no label frequancy threshold is required
        config.LABEL_FREQUENCY_THRESHOLD = 0
        logging.info('reading Movielens new data labels with a threshold of {0}...'.format(config.LABEL_FREQUENCY_THRESHOLD))
        label_path = config.MOVIELENS_DATA_NEW
    elif config.MOVIES_OR_MUSIC == 'movies_newest':
        # the new datasets have no IMDB label file and only uses genres from the movies' file
        # therefore, no label frequancy threshold is required
        config.LABEL_FREQUENCY_THRESHOLD = 0
        logging.info('reading Movielens new data labels with a threshold of {0}...'.format(config.LABEL_FREQUENCY_THRESHOLD))
        label_path = config.MOVIELENS_DATA_NEWEST
    
    with open(label_path, "rb") as label_file:
        
        if config.MOVIES_OR_MUSIC == 'movies_newest':
            label_file_reader = csv.reader(label_file)
        else:
            label_file_reader = label_file
        
        for line in label_file_reader:
            if config.MOVIES_OR_MUSIC == 'movies_newest':
                data = line
            else:
                data = line.split('::')
            config.ITEM_DATA[data[0]] = {'title':data[1], 'labels':data[2].rstrip().split('|')}
    
#     label_frequencies = counter.Counter([label.lower() for labels in label_dict.values() for label in labels])
    label_frequencies = counter.Counter([label.lower() for item_dict in config.ITEM_DATA.values() for label in item_dict['labels']])
    frequent_labels = [label for label, freq in label_frequencies.items() if freq >= config.LABEL_FREQUENCY_THRESHOLD]
    
#     for item_id, tags in label_dict.iteritems():
#         good_labels = [l for l in tags if l.lower() in frequent_labels]
#         if good_labels:
#             config.ITEM_DATA[item_id] = good_labels
    
    for item_id, item_data in config.ITEM_DATA.iteritems():
        good_labels = [label for label in item_data['labels'] if label.lower() in frequent_labels]
        config.ITEM_DATA[item_id]['labels'] = good_labels
    
    # remove items that don't have any frequent labels
    for k, v in config.ITEM_DATA.items():
        if not v['labels']:
            del config.ITEM_DATA[k]
    
    logging.info('done! {0} items have at least one frequent label and will be used for splits'.format(len(config.ITEM_DATA)))


def _readUserIds():
    '''
    read the IDs of users from the rating file
    store them in the global dict USER_DATA
    '''
    
    data_separator = '::'
    
    if not config.MOVIES_OR_MUSIC:
        raise ValueError('MOVIES_OR_MUSIC not set. Cannot continue.')
    elif config.MOVIES_OR_MUSIC == 'movies':
        logging.info('reading Movielens user IDs...')
        file_path = config.MOVIE_RATINGS
    elif config.MOVIES_OR_MUSIC == 'music':
        logging.info('reading LastFM user IDs...')
        file_path = config.ARTIST_RATINGS
    elif config.MOVIES_OR_MUSIC == 'movies_new':
        logging.info('reading Movielens new user IDs...')
        file_path = config.MOVIE_RATINGS_NEW
    elif config.MOVIES_OR_MUSIC == 'movies_newest':
        logging.info('reading Movielens new user IDs...')
        file_path = config.MOVIE_RATINGS_NEWEST
        data_separator = ','
    
    
    with open(file_path, "rb") as f:
        
        for line in f:
            data = line.split(data_separator)
            config.USER_DATA.setdefault(data[0], []).append(data[1])
    
    logging.info('done reading user IDs.')
    

# def getItemTitleById(item_id):
#     '''
#     read the title of a movie or an artist from a data file
#     
#     @param item_id: ID of the item for which to get the title
#     @type item_id: string
#     '''
#     
#     result = None
#     
#     data_separator = '::'
#     
#     if not config.MOVIES_OR_MUSIC:
#         raise ValueError('MOVIES_OR_MUSIC not set. Cannot continue.')
#     elif config.MOVIES_OR_MUSIC == 'movies':
#         title_file_path = config.MOVIELENS_DATA
#     elif config.MOVIES_OR_MUSIC == 'music':
#         title_file_path = config.LASTFM_DATA
#     
#     elif config.MOVIES_OR_MUSIC == 'movies_new':
#         title_file_path = config.MOVIELENS_DATA_NEW
#         data_separator = ','
#     
#     
#     with open(title_file_path, "rb") as title_file:
#         for line in title_file:
#             data = line.split(data_separator)
#             if (data[0] == str(item_id)):
#                 result = data[1].decode('utf8')
#     
#     return result


def _readRatingData():
    '''
    read the Movielens/Last.fm rating data as a list of (user, item, rating) tuples
    checks the labels of each item for a minimal threshold (items that don't have enough content labels are skipped)
    
    stores the program IDs of users/items, not the ones found in the files
    this is needed for Last.fm data, which has non-numeric IDs in the files
    
    _readItemData() and _readUserIds() have to be called before this method execution!
    '''
    
    data_separator = '::'
    
    if (not config.MOVIES_OR_MUSIC) or (not config.ITEM_DATA):
        raise ValueError('one of MOVIES_OR_MUSIC / ITEM_DATA not set. Cannot continue.')
    elif config.MOVIES_OR_MUSIC == 'movies':
        logging.info('reading Movielens data...')
        data_path = config.MOVIE_RATINGS
    elif config.MOVIES_OR_MUSIC == 'music':
        logging.info('reading Last.fm data...')
        data_path = config.ARTIST_RATINGS
    elif config.MOVIES_OR_MUSIC == 'movies_new':
        logging.info('reading Movielens new data...')
        data_path = config.MOVIE_RATINGS_NEW
    elif config.MOVIES_OR_MUSIC == 'movies_newest':
        logging.info('reading Movielens new data...')
        data_path = config.MOVIE_RATINGS_NEWEST
        data_separator = ','
    
    
    data_tuples = []
    
    with open(data_path, "rb") as data_file:
        
#         previous_user = ''
#         user_id_counter = 1
        
        for line in data_file:
            data = line.split(data_separator)
            
#             if data[0] != previous_user:
#                 # new user encountered - record it into the global dict
#                 config.USER_IDS[data[0]] = str(user_id_counter)
#                 previous_user = data[0]
#                 user_id_counter += 1
            
            user_id = data[0]
            item_id = data[1]
            rating = float(data[2].strip('\r\n'))
            
            if item_id in config.ITEM_DATA:
                data_tuples.append((user_id, item_id, rating))
            else:
                logging.warning('skipped item {0} (not enough content labels)'.format(item_id))
    
    logging.info('done!')
    
    return data_tuples


# def getRealItemIdAndTitle(item_id):
#     '''
#     get the real item ID (i.e., one that is used in the dataset files)
#     and the title
#     '''
#     the_real_item_id = None
#     rec_title = None
#     
#     the_real_item_ids = [k for k, v in config.ITEM_IDS.iteritems() if v == item_id]
#     if the_real_item_ids:
#         the_real_item_id = the_real_item_ids[0]
#         rec_title = getItemTitleById(the_real_item_id)
#     
#     return (the_real_item_id, rec_title)


if __name__ == "__main__":
    
    _generateArtistRatings()
    
    
    