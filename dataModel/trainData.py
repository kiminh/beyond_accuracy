'''
Created on 18 Jan 2014

@author: Marius
'''

import logging
import math
import operator

from dataModel.dataMatrix import DataMatrix
from frameworkMetrics import explanationMetrics, novelty
from utils import config


class TrainData(DataMatrix):
    '''
    a class for creating the training data user-item matrix as a CSR sparse matrix,
    as well as user-user and item-item similarity matrices
    '''
    
    def __init__(self, data_path, user_means_path):
        '''
        read data from the split file, create ID-index dictionaries for rows (users) and columns (items)
        create the csc sparse rating matrix
        
        @param data_path: path to a data split file
        @type data_path: string
        
        @param user_means_path: path to the corresponding user means file
        @type user_means_path: string
        '''
        
        super(TrainData, self).__init__(data_path)
        
        logging.info('reading user means from file...')
        self._user_means = {}
        with open(user_means_path, 'rb') as mean_file:
            for line in mean_file:
                data = line.split('\t')
                self._user_means[data[0]] = float(data[1].rstrip('\n'))
        logging.info('done!')
        
        logging.info('computing user-user similarity matrix...')
        self.computeUserSimilarityMatrix()
        logging.info('done! user-user similarity matrix has {0} entries.'.format(self.user_similarity_matrix.nnz))
        
        logging.info('computing item-item similarity matrix...')
        self.computeItemSimilarityMatrix()
        logging.info('done! item-item similarity matrix has {0} entries.'.format(self.item_similarity_matrix.nnz))
        
        logging.info('done initializing training data! rating matrix: {0}; user-user matrix: {1}; item-item matrix: {2}.'\
                     .format(self.rating_matrix.shape, self.user_similarity_matrix.shape, self.item_similarity_matrix.shape))
        
#         for row_index in self._row_indices.values():
#             row = row_matrix.data[row_matrix.indptr[row_index] : row_matrix.indptr[row_index + 1]]
#             if len(row) > 0:
#                 row_mean = np.mean(row)
#                 self.user_means[row_index] = row_mean # store user mean to add it to prediction
#                 row_matrix.data[row_matrix.indptr[row_index] : row_matrix.indptr[row_index + 1]] -= row_mean
#         logging.info('done!')
#         # store the mean-centered matrix in CSC format (required for SVD)
#         self.mean_centered_matrix = row_matrix.tocsc()
#         
#         # initialize the array of item means
#         self.item_means = array.array('f',(0,)*len(self._col_indices))
#         
#         sys.stdout.write('\t now item mean-centering rating matrix... ')
#         item_matrix = user_rating_matrix.tocsc()
#         item_ratings_lost = 0
#         for col_index in self._col_indices.values():
#             col = item_matrix.data[item_matrix.indptr[col_index] : item_matrix.indptr[col_index + 1]]
#             if len(col) > 0:
#                 col_mean = np.mean(col)
#                 self.item_means[col_index] = col_mean # store item mean to add it to prediction
#                 item_matrix.data[item_matrix.indptr[col_index] : item_matrix.indptr[col_index + 1]] -= col_mean
#                 if 0 in item_matrix.data[item_matrix.indptr[col_index] : item_matrix.indptr[col_index + 1]]:
#                     item_ratings_lost += 1
#         sys.stdout.write('done! Lost ratings for '+str(item_ratings_lost)+' items.\n')
#         
#         # store the item-centered matrix
#         self.item_matrix = item_matrix
    
    def computeUserSimilarityMatrix(self):
        '''
        compute the user-to-user similarity matrix
        the rating data has to be row-normalized
        stores the result as a CSR sparse matrix
        '''
        
        row_matrix = self.rating_matrix.copy()
        
        for row_index in self._row_indices.values():
            row = row_matrix.data[row_matrix.indptr[row_index] : row_matrix.indptr[row_index + 1]]
            if len(row) > 0:
                row_length = math.sqrt(sum(row[i]*row[i] for i in range(len(row))))
                if row_length > 0:
                    row_matrix.data[row_matrix.indptr[row_index] : row_matrix.indptr[row_index + 1]] /= row_length
        
        self.user_similarity_matrix = row_matrix.dot(row_matrix.T)
    
    
    def computeItemSimilarityMatrix(self):
        '''
        compute the item-to-item similarity matrix
        the rating data has to be column-normalized
        stores the result as a CSC sparse matrix
        '''
        
        col_matrix = self.rating_matrix.tocsc()
        
        for col_index in self._col_indices.values():
            col = col_matrix.data[col_matrix.indptr[col_index] : col_matrix.indptr[col_index + 1]]
            if len(col) > 0:
                col_length = math.sqrt(sum(col[i]*col[i] for i in range(len(col))))
                if col_length > 0:
                    col_matrix.data[col_matrix.indptr[col_index] : col_matrix.indptr[col_index + 1]] /= col_length
        
        self.item_similarity_matrix = col_matrix.T.dot(col_matrix)
    
    
    def getUserNeighboursByIndex(self, user_index, candidate_neighbour_indices):
        '''
        @param user_index: index of the user
        @type user_index: string
        
        @param candidate_neighbour_indices: indices of users to get the neighbours from (None means all users are candidates)
        @type candidate_neighbour_indices: list
        
        @return: (neighbour_index, neighbour_similarity) tuples for top K user neighbours from the user-user similarity matrix
        '''
        
        K = config.NEIGHBOURHOOD_SIZE
        user_row = self.user_similarity_matrix.getrow(user_index)
        
        neighbour_tuples = [i for i in sorted(zip(user_row.indices, user_row.data), key=operator.itemgetter(1), reverse=True) if i[0] != user_index][:K]
        
        if candidate_neighbour_indices:
            return [i for i in neighbour_tuples if i[0] in candidate_neighbour_indices]
        else:
            return neighbour_tuples
    
    
    def getItemNeighboursByIndex(self, item_index, candidate_neighbour_indices):
        '''
        @param item_index: index of the item
        @type item_index: string
        
        @param candidate_neighbour_indices: indices of items to get the neighbours from (None means all items are candidates)
        @type candidate_neighbour_indices: list
        
        @return: (neighbour_index, neighbour_similarity) tuples for top K item neighbours from the item-item similarity matrix
        '''
        
        K = config.NEIGHBOURHOOD_SIZE
        item_row = self.item_similarity_matrix.getrow(item_index)
        
        if candidate_neighbour_indices:
            return [i for i in sorted(zip(item_row.indices, item_row.data), key=operator.itemgetter(1), reverse=True) if i[0] in candidate_neighbour_indices][:K]
        else:
            return [i for i in sorted(zip(item_row.indices, item_row.data), key=operator.itemgetter(1), reverse=True) if i[0] != item_index][:K]
    
    
    def getFactorBasedRecommendations(self, user_id, Q, evaluation_item_ids):
        '''
        predict the rating approximation (using the SVD decomposition) for the evaluation_item_ids and sort them according to prediction_score (descending order)
        
        @param user_id: IDs of the user for whom predictions need to be made
        @type user_id: int
        
        @param Q: indices of items to get the neighbours from (default is None, which means all items are candidates)
        @type Q: CSC sparse matrix
        
        @param evaluation_item_ids: IDs of the items for which predictions need to be made
        @type evaluation_item_ids: list
        
        @return: list of (item_id, prediction_score) tuples, sorted in descending order
        '''
        
        predictions = []
        
        user_index = self.getUserIndex(user_id)
        user_ratings = self.rating_matrix.getrow(user_index)
        
        for item_id in evaluation_item_ids:
            item_index = self.getItemIndex(item_id)
            prediction = user_ratings.dot(Q.T).dot(Q[:, item_index])[0]
            predictions.append((item_id, prediction))
        
        return sorted(predictions, key=operator.itemgetter(1), reverse=True)
    
    
    def getUserBasedRecommendations(self, user_id, evaluation_item_ids, rating_prediction_method, verbose=False):
        '''
        predict the ratings using user-based CF for the evaluation_item_ids and sort them according to prediction_score (descending order)
        three options of the weighted rating prediction:
        - classic: the handbook version - neighbours are computed only among the users who rated the target item 
        - self_damping: neighbours are computed among all users; all neighbour similarities go into the denominator to dampen rare item prediction scores
        - non_normalized: from [Cremonesi et al., 2009] - neighbours selected like in 'classic',
          but no denominator is used in the prediction f-la, therefore it has a damping effect for rare items (i.e., items with few overlapping neighbours)  
        
        @param user_id: IDs of the user for whom predictions need to be made
        @type user_id: int
        
        @param evaluation_item_ids: IDs of the items for which predictions need to be made
        @type evaluation_item_ids: list
        
        @param rating_prediction_method: a switch to select the weighted rating prediction method 
        @type rating_prediction_method: {classic, self_damping, non_normalized}
        
        @param verbose: a flag to enable detailed printing of prediction score computation (default is False)
        @type verbose: bool
        
        @return: list of (item_id, prediction_score) tuples, sorted in descending order
        '''
        
        predictions = []
        user_index = self.getUserIndex(user_id)
        if verbose:
            logging.info('\n--------------- \ncomputing UB {0} recommendations for items {1} \n---------------'.format(rating_prediction_method, evaluation_item_ids))
        
        if rating_prediction_method == 'self_damping':
            neighbour_users = self.getUserNeighboursByIndex(user_index, None)
        
        for item_id in evaluation_item_ids:
            
            item_index = self.getItemIndex(item_id)
            
            if rating_prediction_method == 'classic' or rating_prediction_method == 'non_normalized':
                candidate_neighbour_indices = set(self.getItemProfileByIndex(item_index))
                neighbour_users = self.getUserNeighboursByIndex(user_index, candidate_neighbour_indices)
                
            elif rating_prediction_method == 'self_damping':
                pass
                
            else:
                raise ValueError('Wrong rating_prediction_method. Choose between classic, non_normalized, and self_damping.')
            
            numerator_string = ''
            denominator_string = ''
            
            numerator = 0.0
            denominator = 0.0
            
            for neighbour_index, neighbour_similarity in neighbour_users:
                
                neighbour_rating = self.rating_matrix[neighbour_index, item_index]
                numerator += neighbour_similarity * neighbour_rating
                if verbose:
                    numerator_string += '('+str(neighbour_similarity)+' x '+str(neighbour_rating)+') + '
                
                if rating_prediction_method == 'classic' or rating_prediction_method == 'self_damping':
                    denominator += abs(neighbour_similarity)
                    if verbose:
                        denominator_string += str(abs(neighbour_similarity))+' + '
            
            if rating_prediction_method == 'classic' or rating_prediction_method == 'self_damping':
                if denominator == 0.0:
                    continue
#                     prediction = (self.getItemId(item_index), -sys.maxint)
                    
                else:
                    prediction = (self.getItemId(item_index), self._user_means[user_id] + (numerator / denominator))
                
                predictions.append(prediction)
                if verbose:
                    logging.info('{0} / {1} = {2}'.format(numerator_string, denominator_string, prediction))
            
            elif rating_prediction_method == 'non_normalized':
                prediction = (self.getItemId(item_index), self._user_means[user_id] + numerator)
                predictions.append(prediction)
                if verbose:
                    logging.info('{0} = {1}'.format(numerator_string, prediction))
        
        return sorted(predictions, key=operator.itemgetter(1), reverse=True)
    
    
    def getItemBasedRecommendations(self, user_id, evaluation_item_ids, rating_prediction_method, verbose=False):
        '''
        predict the ratings using item-based CF for the evaluation_item_ids and sort them according to prediction_score (descending order)
        three options of the weighted rating prediction:
        - classic: the handbook version - item neighbours are computed only among items the target user has rated
        - self_damping: item neighbours are computed among all items; all neighbour similarities go into the denominator to dampen rare item prediction scores
        - non_normalized: from [Cremonesi et al., 2009] - neighbours selected like in 'classic',
          but no denominator is used in the prediction f-la, therefore it has a damping effect for rare items (i.e., items with few overlapping neighbours)  
        
        @param user_id: IDs of the user for whom predictions need to be made
        @type user_id: int
        
        @param evaluation_item_ids: IDs of the items for which predictions need to be made
        @type evaluation_item_ids: list
        
        @param rating_prediction_method: a switch to select the weighted rating prediction method 
        @type rating_prediction_method: {classic, self_damping, non_normalized}
        
        @param verbose: a flag to enable detailed printing of prediction score computation (default is False)
        @type verbose: bool
        
        @return: list of (item_id, prediction_score) tuples, sorted in descending order
        '''
        
        predictions = []
        user_index = self.getUserIndex(user_id)
        
        for item_id in evaluation_item_ids:
            
            item_index = self.getItemIndex(item_id)
            
            if rating_prediction_method == 'classic' or rating_prediction_method == 'non_normalized':
                candidate_neighbour_indices = set(self.getUserProfileByIndex(user_index))
                neighbour_items = self.getItemNeighboursByIndex(item_index, candidate_neighbour_indices)
                
            elif rating_prediction_method == 'self_damping':
                neighbour_items = self.getItemNeighboursByIndex(item_index, None)
                
            else:
                raise ValueError('Wrong rating_prediction_method. Choose between classic, non_normalized, and self_damping.')
            
            numerator_string = ''
            denominator_string = ''
            
            numerator = 0.0
            denominator = 0.0
            
            for neighbour_index, neighbour_item_similarity in neighbour_items:
                
                neighbour_item_rating = self.rating_matrix[user_index, neighbour_index]
                numerator += neighbour_item_similarity * neighbour_item_rating
                if verbose:
                    numerator_string += '('+str(neighbour_item_similarity)+' x '+str(neighbour_item_rating)+') + '
                
                if rating_prediction_method == 'classic' or rating_prediction_method == 'self_damping':
                    denominator += abs(neighbour_item_similarity)
                    if verbose:
                        denominator_string += str(abs(neighbour_item_similarity))+' + '
            
            if rating_prediction_method == 'classic' or rating_prediction_method == 'self_damping':
                if denominator == 0.0:
                    continue
#                     prediction = (self.getItemId(item_index), -sys.maxint)
                    
                else:
                    prediction = (self.getItemId(item_index), self._user_means[user_id] + (numerator / denominator))
                
                predictions.append(prediction)
                if verbose:
                    logging.info('{0} / {1} = {2}'.format(numerator_string, denominator_string, prediction))
                
            elif rating_prediction_method == 'non_normalized':
                prediction = (self.getItemId(item_index), self._user_means[user_id] + numerator)
                predictions.append(prediction)
                
                if verbose:
                    logging.info('{0} = {1}'.format(numerator_string, prediction))
        
        return sorted(predictions, key=operator.itemgetter(1), reverse=True)
    
    
    
    def getCandidateOpinions(self, user_index, item_index, neighbour_users, extended_candidates, mean_center, verbose):
        '''
        get the explanation partner - the most similar of neighbours who like the target item
        then get candidate opinions - items the users agree on
        opinions are stored as a dict {(item_index, opinion): neighbour_sim}, where opinion = dislike/neutral/like
        '''
        
        user_id = self.getUserId(user_index)
        
#         if verbose:
#             logging.info('going through neighbours {0}'.format(neighbour_users))
        
        candidate_opinions = {}
        # neighbour_users must be sorted by decreasing similarity
        for neighbour_index, neighbour_similarity in neighbour_users:
            
            neighbour_id = self.getUserId(neighbour_index)
            if mean_center:
                neighbour_rating = self.rating_matrix[neighbour_index, item_index]
            else:
                neighbour_rating = round(self._user_means[neighbour_id] + self.rating_matrix[neighbour_index, item_index], 1)
            
            if verbose:
                logging.info('considering a neighbour ID{0} with similarity {1}, who rated the item as {2}'.format(neighbour_id, neighbour_similarity, neighbour_rating))
            
            
            if (neighbour_rating > 3.0 and not mean_center) or (neighbour_rating > 0 and mean_center):
                
                if verbose:
                    logging.info('\t and he is indeed our explanation partner!')
                
                partner_profile = self.getUserProfileByIndex(neighbour_index)
                for item_ind in partner_profile:
                    
                    user_rating = self.rating_matrix[user_index, item_ind]
                    # only consider items the target user has rated
                    if user_rating == 0.0:
                        continue
                    
                    if mean_center:
                        
                        # if needed, skip dislikes
                        if config.NO_DISLIKES and user_rating < 0:
                            continue
                        
                        partner_rating = self.rating_matrix[neighbour_index, item_ind]
                        
                        if partner_rating > 0 and user_rating > 0:
                            if ( (item_ind,'like') in candidate_opinions ) and ( candidate_opinions[(item_ind,'like')] >= neighbour_similarity ):
                                # skip item opinions that are already suggested by higher similarity neighbours
                                continue
                            candidate_opinions[(item_ind,'like')] = neighbour_similarity
                            
                        elif partner_rating < 0 and user_rating < 0:
                            if ( (item_ind,'dislike') in candidate_opinions ) and ( candidate_opinions[(item_ind,'dislike')] >= neighbour_similarity ):
                                # skip item opinions that are already suggested by higher similarity neighbours
                                continue
                            candidate_opinions[(item_ind,'dislike')] = neighbour_similarity
                        # opinions for rating==0.0 should never be needed (these ratings would not appear in the rating matrix anyway)
                        
                    else:
                        
                        user_rating = round(float(self._user_means[user_id] + user_rating), 1)
                        # if needed, skip dislikes/neutrals
                        if config.NO_DISLIKES and user_rating <= 3.0:
                            continue
                        
                        partner_rating = round(float(self._user_means[neighbour_id] + self.rating_matrix[neighbour_index, item_ind]), 1)
                        
                        if partner_rating > 3.0 and user_rating > 3.0:
                            if ( (item_ind,'like') in candidate_opinions ) and ( candidate_opinions[(item_ind,'like')] >= neighbour_similarity ):
                                # skip item opinions that are already suggested by higher similarity neighbours
                                continue
                            candidate_opinions[(item_ind,'like')] = neighbour_similarity
                            
                        elif partner_rating < 3.0 and user_rating < 3.0:
                            if ( (item_ind,'dislike') in candidate_opinions ) and ( candidate_opinions[(item_ind,'dislike')] >= neighbour_similarity ):
                                # skip item opinions that are already suggested by higher similarity neighbours
                                continue
                            candidate_opinions[(item_ind,'dislike')] = neighbour_similarity
                            
                        elif partner_rating == 3.0 and user_rating == 3.0:
                            if ( (item_ind,'neutral') in candidate_opinions ) and ( candidate_opinions[(item_ind,'neutral')] >= neighbour_similarity ):
                                # skip item opinions that are already suggested by higher similarity neighbours
                                continue
                            candidate_opinions[(item_ind,'neutral')] = neighbour_similarity
                
                # if extended_candidate_search is False, return candidates as soon as the first explanation partner is found
                if candidate_opinions and not extended_candidates:
                    return candidate_opinions
                # else, continue adding candidate opinions for all neighbours
            
        return candidate_opinions
    
    
    def generateExplanations(self, user_index, item_id, objective_metric, extended_candidates, acc_filter, mean_center=False, verbose=False):
        '''
        generate explanations in a form of 'you liked A and B, therefore we recommend C' for the user-based k-NN method
        the explanation rule is stored as a list of tuples [(item_id, opinion)], where opinion=dislike/neutral/like
            omitting the target item (item_id, like)
        
        the rules are constructed in a greedy fashion, maximizing the objective metric
        which can be any of the config.RULE_METRICS = {accuracy, discounted_accuracy, lift, odds_ratio, novelty, surprise}
        ties are always broken by coverage
        '''
        
        if verbose:
            logging.info('Generating explanations for item {0} using objective {1}'.format(item_id, objective_metric))
        
        if (not config.RULE_METRICS) or (objective_metric not in config.RULE_METRICS):
            raise ValueError('RULE_METRICS not set properly. Cannot continue.')
        
        item_index = self.getItemIndex(item_id)
        
        candidate_neighbour_indices = set(self.getItemProfileByIndex(item_index))
        neighbour_users = self.getUserNeighboursByIndex(user_index, candidate_neighbour_indices)
        candidate_opinions = self.getCandidateOpinions(user_index, item_index, neighbour_users, extended_candidates, mean_center, verbose)
        
        # for a user-based k-NN, there always has to exist an explanation partner
        # UNLESS, there are no neighbours who LIKED the item (e.g., all may have disliked it)
        if not candidate_opinions:
            if verbose:
                logging.info('couldn\'t find an explanation!')
            return None, None
        
        if verbose:
            logging.info('going through the candidate opinions {0}'.format([(self.getItemId(i),op,sim) for (i, op), sim in candidate_opinions.iteritems()]))
        
        # initialize the result variables
        explanation_rule = []
        explanation_rule_metrics = {}
        for m in config.RULE_METRICS:
            explanation_rule_metrics[m] = 0.0
        if extended_candidates:
            # in case of extended candidate search, keep the sim-weighted objective metric separate
            explanation_rule_metrics[objective_metric+'_ex'] = 0.0
        
        # main loop of the greedy algorithm
        while len(candidate_opinions) > 0:
            
            # a dict to store all candidate additions of the current iteration
            # {(item,opinion): {metric_name: metric_value})
            candidate_antecedent_additions = {}
            for (candidate_index, opinion), neighbour_sim in candidate_opinions.iteritems():
                
                candidate_id = self.getItemId(candidate_index)
                antecedent_tmp = explanation_rule + [(candidate_id, opinion)]
                rule_metrics = explanationMetrics.getRuleMetrics(self, antecedent_tmp, (item_id,'like'), verbose)
                
                # if item-weighted metrics are included, update their values accordingly
                if 'novelty_accuracy' in rule_metrics:
                    item_novelty = novelty._getItemNovelty(self, candidate_id)
                    rule_accuracy = rule_metrics['accuracy']
                    rule_metrics['novelty_accuracy'] = (0.5 * rule_accuracy) + (0.5 * item_novelty)
                    
                if 'similarity_accuracy' in rule_metrics:
                    a = set(config.ITEM_DATA[item_id]['labels'])
                    b = set(config.ITEM_DATA[candidate_id]['labels'])
                    c = a.intersection(b)
                    item_similarity = float(len(c)) / (len(a) + len(b) - len(c))
                    rule_accuracy = rule_metrics['accuracy']
                    rule_metrics['similarity_accuracy'] = (0.5 * rule_accuracy) + (0.5 * item_similarity)
                
                # if the extended candidate search is True, weight the objective metric by the neighbours' similarities
                if extended_candidates:
                    rule_metrics[objective_metric+'_ex'] = rule_metrics[objective_metric] * neighbour_sim
                
                candidate_antecedent_additions[(candidate_id, opinion)] = rule_metrics
                
                if verbose:
                    logging.info('current candidate opinion on item {0} forms the antecedent {1}'.format(candidate_id, antecedent_tmp))
                    logging.info('\t the rule\'s metrics are {0}'.format(candidate_antecedent_additions[(candidate_id, opinion)]))
                
            
            # sort the rules by the objective metric, pick the best ( (item,opinion), rule_dict ) tuple
            sort_metric = objective_metric
            if extended_candidates:
                sort_metric = objective_metric+'_ex'
                
            best_candidate_tuple = sorted(candidate_antecedent_additions.iteritems(), key=lambda x: x[1][sort_metric], reverse=True)[0]
            
            if verbose:
                logging.info('best candidate opinion is {0}'.format(best_candidate_tuple[0]))
                logging.info('\t its metrics are:{0})'.format(best_candidate_tuple[1]))
            
            
            # terminate the greedy algorithm when there is no objective metric (or accuracy) improvement
            if acc_filter and (best_candidate_tuple[1]['accuracy'] <= explanation_rule_metrics['accuracy']):
                if verbose:
                    logging.info('no accuracy improvement, returning {0}'.format(explanation_rule))
                
                return explanation_rule, explanation_rule_metrics
                
            elif best_candidate_tuple[1][sort_metric] <= explanation_rule_metrics[sort_metric]:
                if verbose:
                    logging.info('no {0} improvement, returning {1}'.format(sort_metric, explanation_rule))
                
                return explanation_rule, explanation_rule_metrics
            
            # if the search termination conditions are not matched:
            # add the best candidate opinion to the rule,
            explanation_rule.append(best_candidate_tuple[0])
            # record the current metrics of the rule,
            for metric_name, metric_value in best_candidate_tuple[1].iteritems():
                explanation_rule_metrics[metric_name] = metric_value
            # remove the added opinion from the candidates,
            assert (self.getItemIndex(best_candidate_tuple[0][0]), best_candidate_tuple[0][1]) in candidate_opinions
            del candidate_opinions[(self.getItemIndex(best_candidate_tuple[0][0]), best_candidate_tuple[0][1])]
            # and keep looping
            
            if verbose:
                logging.info('iteration complete, the new rule is {0}'.format(explanation_rule))
            
            
        
        if explanation_rule:
            if verbose:
                logging.info('done searching among candidate opinions, returning {0} with metrics {1}'.format(explanation_rule, explanation_rule_metrics))
            return explanation_rule, explanation_rule_metrics
        else:
            if verbose:
                logging.info('done searching among candidate opinions, no explanations found')
            return None, None
    
    
        