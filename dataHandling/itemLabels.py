'''
Created on 13 Jan 2015

a module for getting item labels using the external APIs

@author: mkaminskas
'''
import ast
import logging
import operator
import os
import sys
from time import time

from utils import config


req_version = (2,7)
cur_version = sys.version_info

if cur_version >= req_version:
    
#     import musicbrainz2.webservice as ws
#     import pylast
    import imdb
    
    
    '''
    get IMDB plot keywords for a movie
    '''
    def _getIMDBDataByTitle(title):
        
        try:
            ia = imdb.IMDb()
            movie = ia.search_movie(title)[0]
            ia.update(movie, 'keywords')
            return movie['keywords']
        except Exception, e:
            print e,': could not get keywords for the movie',title
            return []
    
    
    '''
    get the artist names and if needed their tags using the MusicBrainz web service
    if tags are found, return top 5 by count (or less if there aren't 5)
    '''
    def _getMusicBrainzDataByIndex(data_matrix, item_index, tags=False):
        pass
#         
#         item_id = data_matrix.getItemId(item_index)
#         if item_id == -1:
#             print 'Could not find the id of artist '+str(item_index)
#         else:
#             # get artist name from MusicBrainz. 1s delay needed to avoid service error
#             time.sleep(1)
#             q = ws.Query()
#             if tags:
#                 include = ws.ArtistIncludes(tags=True)
#             else:
#                 include = None
#               
#             try:
#                 artist = q.getArtistById(item_id, include)
#                 artist_tags = [(tag.getValue(), tag.getCount()) for tag in artist.getTags()]
#                 return artist.getName(), sorted(artist_tags, key=operator.itemgetter(1), reverse=True)[:min(len(artist_tags),5)]
#               
#             except ws.WebServiceError, e:
#                   
#                 print 'Could not find the title of artist '+str(item_id)+': '+str(e)
#                 return None, None
    
    '''
    read top LastFM tags for an artist
    '''
    def _getLastFMDataById(item_id):
        pass
#         
#         time.sleep(1)
#         
#         API_KEY = "e397a21f9334aaa9233e8d38ea2e6500" # this is a sample key
#         network = pylast.LastFMNetwork(api_key=API_KEY)
#         
#         try:
#             artist = network.get_artist_by_mbid(item_id)
#             topItems = artist.get_top_tags(limit=10)
#             if topItems:
#                 return artist.get_name(), [topItem.item.get_name().lower() for topItem in topItems]
#             else:
#                 return None, None
#         except Exception, e:
#             print e
#             return None, None


'''
write the file of movies and their content labels (genres + IMDB keywords)
'''
def _generateMovieLabels(dataset='old'):
    '''
    dataset = old | new | big
    although 'new' and 'big' setups need updating the IMDB parser
    '''
    
    t = time()
    
    if dataset == 'old':
        logging.info('generating labels for the 1M Movielens data...')
        source_data_path = config.MOVIELENS_DATA
        destination_data_path = config.MOVIE_FILE_IMDB
        data_separator = '::'
    elif dataset == 'new':
        logging.info('generating labels for the latest Movielens data...')
        source_data_path = config.MOVIELENS_DATA_NEW
        destination_data_path = config.MOVIE_FILE_IMDB
        data_separator = ','
    elif dataset == 'big':
        logging.info('generating labels for the 20M Movielens data...')
        source_data_path = ''
        destination_data_path = ''
        data_separator = ','
    else:
        raise ValueError('Wrong type of dataset entered.')
    
    
    processed_movies = set()
    # before reading the data, need to construct a list of already processed artists
    with open(destination_data_path,'rb') as movies:
        for line in movies:
            data = line.split('::')
            processed_movies.add(data[0])
    
    with open(source_data_path, 'rb') as f:
        with open(destination_data_path,'a') as movie_file:
            
            for line in f:
                data = line.split(data_separator)
                movie_id = data[0]
                
                if (movie_id not in processed_movies):
                    
                    movie_title = data[1].decode('utf8')
                    movie_genres = data[2].rstrip().split('|')
                    movie_keywords = _getIMDBDataByTitle(movie_title)
                    
                    movie_labels = "|".join(movie_genres + movie_keywords)
                    
                    movie_file.write(movie_id+'::'+movie_title.encode('utf8')+'::'+movie_labels.encode('utf8')+'\n')
                    processed_movies.add(movie_id)
                    
                    print 'done with movie',movie_title
                    print 'number of unique movies:',len(processed_movies)
                
    print("movie data generated in %0.3fs." % (time() - t))


'''
write the file of unique artists and their LastFM tags
the processed_artists set needed to make this method re-runnable 
'''
def _generateArtistLabels():
    t = time()
    
    unidentified_artists = set()
    processed_artists = set()
    # before reading the data, need to construct a list of already processed artists
    with open(config.ARTIST_FILE_LASTFM,'rb') as artists:
        for line in artists:
            data = line.split('::')
            processed_artists.add(data[0])
    
    with open(config.LASTFM_DATA, 'rb') as f:
        with open(config.ARTIST_FILE_LASTFM,'a') as artist_file:
        
            for line in f:
                data = line.split('\t')
                artist_id = data[2]
                
                if artist_id != '':
                    
                    # check if the artist is not yet processed
                    if (artist_id not in processed_artists) and (artist_id not in unidentified_artists):
                        
                        artist_name, artist_tags = _getLastFMDataById(artist_id)
                        # only record artists that have at least 3 tags
                        if artist_name and len(artist_tags) >= 3:
                            
                            labels = "|".join(artist_tags)
                            artist_file.write(artist_id+'::'+artist_name.encode('utf8')+'::'+labels.encode('utf8')+'\n')
                            processed_artists.add(artist_id)
                            
                        else:
                            unidentified_artists.add(artist_id)
                    
                    print 'number of unique artists:',len(processed_artists)
                
    print("artist data generated in %0.3fs." % (time() - t))


def getMovieSynopsisForSurvey(iteration):
    '''
    get the genres and plot synopsis for movies used in the user study
    '''
    movies = {}
#     with open(os.path.join(config.PACKAGE_DIR, '../survey/mov.txt'),'rb') as f:
#         movies = ast.literal_eval(f.readline())
    
    with open(os.path.join(config.PACKAGE_DIR, '../survey/survey'+str(iteration)+'.dat'),'rb') as survey_file:
        for line in survey_file:
             
            if '"' not in line:
                continue
             
            titles = line.split('"')[1::2]
            for title in titles:
                if title not in movies:
                    
                    try:
                        ia = imdb.IMDb()
                        movie = ia.search_movie(title)[0]
                        movie = ia.get_movie(movie.movieID)
                        movies[title] = {'genres':movie.get('genres'), 'plot':movie.get('plot outline')}
                        print 'processed',len(movies),'movies'
                        
                    except Exception, e:
                        print 'could not get the data for movie',title
                        print e
    
    # writing the sorted dict to a file
    with open(os.path.join(config.PACKAGE_DIR, '../survey/movies'+str(iteration)+'.dat'),'wb') as movie_file:
        for key in sorted(movies):
            
            if movies[key]['genres'] is None:
                genres = 'None'
            else:
                genres = " | ".join(movies[key]['genres'])
            
            if movies[key]['plot'] is None:
                plot = 'None'
            else:
                plot = movies[key]['plot']
            
            movie_file.write(key+' | '+genres.encode('utf8')+'\nsynopsis: '+plot+'\n-------------------------\n')


if __name__ == "__main__":
    
#     _generateMovieLabels()
#     _generateArtistLabels()
    for i in range(1,6):
        getMovieSynopsisForSurvey(i)
    