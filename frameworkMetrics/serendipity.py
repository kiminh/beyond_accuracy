'''
Created on 12 Feb 2015

@author: mkaminskas
'''
import math
import operator

from utils import config


def getListSerendipity(training_data, user_index, item_list, method, exclude_item_itself=False):
    '''
    the average list serendipity
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param user_index: index of the user for whom serendipity needs to be computed
    @type user_index: string
    
    @param item_list: (id, score) tuples of items
    @type item_list: list
    
    @param method: the method used to compute list serendipity {cooccurrence, content}
    @type method: string
    
    @param exclude_item_itself: the switch allowing to exclude the item in question from the user's profile (if it's there) to avoid getting 0.0 surprise
    @type exclude_item_itself: bool
    
    @return: serendipity of the item list with respect to the user
    @rtype: float
    '''
    
    serendipity_sum = 0.0
    
    if method == 'sur_r_n':
        max_freq = training_data.getMaxItemCooccurrenceValue(item_list, user_index)
    else:
        max_freq = None
    
    for item_id, _ in item_list:
        serendipity_sum += _getItemSerendipity(training_data, user_index, item_id, max_freq, method, exclude_item_itself)
    
    return serendipity_sum / len(item_list)



def getMinMaxSerendipity(training_data, user_index, item_list, method, exclude_item_itself=False):
    '''
    the min and max values of item list serendipity
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param item_list: (id, score) tuples of items
    @type item_list: list
    
    @param exclude_item_itself: the switch allowing to exclude the item in question from the user's profile (if it's there) to avoid getting 0.0 surprise
    @type exclude_item_itself: bool
    
    @return: min and max serendipities of the item list
    @rtype: tuple
    '''
    
    if method == 'sur_r_n':
        max_freq = training_data.getMaxItemCooccurrenceValue(item_list, user_index)
    else:
        max_freq = None
    
    tmp_serendipities = []
    for item_id, _ in item_list:
        tmp_serendipities.append((item_id, _getItemSerendipity(training_data, user_index, item_id, max_freq, method, exclude_item_itself)))
    
    sorted_serendipities = sorted(tmp_serendipities, key=operator.itemgetter(1))
    return sorted_serendipities[0][1], sorted_serendipities[-1][1]



def _getItemSerendipity(training_data, user_index, item_id, max_freq, method, exclude_item_itself=False):
    '''
    serendipity of an item, computed as min{j in P} dist(i,j)
    where P is the set of items rated by the user and dist(i,j) can be computed using item co-occurrence in user profiles or content items' label distance
    
    the normalized version of co-occ is introduced to prevent rare items from getting very high PMI scores (and consequently low distance values)
    this behaviour causes RS approaches that produce rare recommendations to get low serendipity values compared to more popularity-geared recommenders.
    
    
    @param training_data: the training data object
    @type training_data: trainData.TrainData
    
    @param user_index: index of the user for whom serendipity needs to be computed
    @type user_index: string
    
    @param item_id: ID of item whose serendipity we need to compute
    @type item_id: string
    
    @param method: the method used to compute item serendipity {sur_r, sur_r_n, sur_c}
    @type method: string
    
    @param exclude_item_itself: the switch allowing to exclude the item in question from the user's profile (if it's there) to avoid getting 0.0 surprise
    @type exclude_item_itself: bool
    
    @return: serendipity of the item with respect to the user
    @rtype: float
    '''
    
    item_index = training_data.getItemIndex(item_id)
    user_rated_items = training_data.getUserProfileByIndex(user_index)
    distances = []
    
    for i_index in user_rated_items:
        
        # if exclude_item_itself, skip the item itself not to get 0.0
        if exclude_item_itself and (i_index == item_index):
            continue
        
        i_id = training_data.getItemId(i_index)
        
        if method == 'sur_r':
            if (item_index, i_index) in config.COOCC_SERENDIPITIES:
                pmi = config.COOCC_SERENDIPITIES[(item_index, i_index)]
                
            elif (i_index, item_index) in config.COOCC_SERENDIPITIES:
                pmi = config.COOCC_SERENDIPITIES[(i_index, item_index)]
                
            else:
                pmi, _ = _getPMI(training_data, item_index, i_index)
                config.COOCC_SERENDIPITIES[(item_index, i_index)] = pmi
            
            # must convert PMI in [-1,1] to distance in [0,1]: 1 - [(PMI - min) / (max - min)] = (1 - PMI) / 2
            coocc_distance = (1.0 - pmi) / 2
            distances.append(coocc_distance)
            
            
#             print 'item',item_id,'pmi against ',i_id,'is',pmi,'so we get',coocc_distance
            
            
        elif method == 'sur_r_n':
            pmi, freq = _getPMI(training_data, item_index, i_index)
            config.COOCC_SERENDIPITIES[(item_index, i_index)] = pmi
            
            # must convert PMI in [-1,1] to distance in [0,1]: 1 - [(PMI - min) / (max - min)] = (1 - PMI) / 2
            coocc_distance = (1.0 - (pmi * freq / max_freq)) / 2
            distances.append(coocc_distance)
            
#             print 'item',item_id,'pmi against ',i_id,'is',pmi
#             print 'then we normalize by',freq,'/',max_freq,'so we get',coocc_distance
            
            
        elif method == 'sur_c':
            if (item_index, i_index) in config.CONT_SERENDIPITIES:
                distances.append(config.CONT_SERENDIPITIES[(item_index, i_index)])
                
            elif (i_index, item_index) in config.CONT_SERENDIPITIES:
                distances.append(config.CONT_SERENDIPITIES[(i_index, item_index)])
                
            else:
                a = set(config.ITEM_DATA[item_id]['labels'])
                b = set(config.ITEM_DATA[i_id]['labels'])
                c = a.intersection(b)
                content_distance = 1 - ( float(len(c)) / (len(a) + len(b) - len(c)) )
                distances.append(content_distance)
                config.CONT_SERENDIPITIES[(item_index, i_index)] = content_distance
            
            
        else:
            raise ValueError('Wrong serendipity method entered. Choose between sur_r, sur_r_n, and sur_c.')
    
#     print '\n##########################',min(distances),'###################'
    
    return min(distances)
    
    

def _getPMI(training_data, item_index1, item_index2):
    '''
    the weighted point-wise mutual information of the target item with respect to item in user's profile:
    PMI(x, y) = freq(x,y) * log [p(x,y) / p(x)*p(y)]
    
    PMI is normalized dividing by -log[p(x,y) to get the normalized value between -1.0 and 1.0
    
    -1.0 -> the two items never occur together
    0.0 -> item distributions are independent
    1.0 -> the two items always occur together
    
    item probabilities p(x), p(y) and p(x,y) can be estimated across all dataset,
    or only among user's neighbours (currently not implemented)
    '''
    
    pmi = 0.0
    N = training_data.getTotalUserNumber()
    
    raters_x = set(training_data.getItemProfileByIndex(item_index1))
    freq_x = len(raters_x)
    
    raters_y = set(training_data.getItemProfileByIndex(item_index2))
    freq_y = len(raters_y)
    
    freq_x_y = float(len(raters_x & raters_y))
    
    if freq_x_y == 0.0:
        pmi = -1.0
        
    else:
        numerator = math.log( (N * freq_x_y) / (freq_x * freq_y), 2 )
        denominator = - math.log(freq_x_y / N, 2)
        pmi = numerator / denominator
    
    return pmi, freq_x_y

