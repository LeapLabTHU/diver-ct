import nltk
import numpy as np

import torch
from tqdm import tqdm
from typing import List


from fast_bleu import SelfBLEU

import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
red_teaming_directory = current_directory.split('red_teaming')[0] + 'red_teaming'

os.chdir(red_teaming_directory)
sys.path.insert(0, red_teaming_directory)

from red_teaming.evaluation.metrics.multiset_distance import MultisetDistances
from red_teaming.evaluation.metrics.vendi_score import text_utils

class SelfScores:
    def __init__(self, queries_tokenized: list = None , responses_tokenized: list = None, queries_untokenized: list = None, responses_untokenized: list = None, min_ngram: int = 2, max_ngram: int = 5, device: str = 'cpu'):
        """
        Initialize SelfScores with tokenized queries and responses.

        Args:
            queries_tokenized (list): List of tokenized queries. Can be None.
            responses_tokenized (list): List of tokenized responses. Can be None.
            queries_untokenized (list): List of untokenized queries. Can be None.
            responses_untokenized (list): List of untokenized responses. Can be None.
            max_ngram (int): Maximum size of n-gram for SelfBLEU score.
            device (str): Device for computation (e.g., 'cpu', 'cuda').
        """
        self.queries_tokenized = queries_tokenized
        self.responses_tokenized = responses_tokenized
        self.queries_untokenized = queries_untokenized
        self.responses_untokenized = responses_untokenized
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.device = device
        self.vendi_model = None # cache the model and tokenizer
        self.vendi_tokenizer = None
    
    def get_score(self):
        """
        Get the self scores for both queries and responses using SelfBLEU and VENDI.

        Returns:
            dict: Dictionary containing self scores for both queries and responses. The scores have both mean and std
            eg:
                {'queries': {'SelfBLEU': float, 'VENDI': {'bert_vendi': [float], 'semantic_vendi': [float], 'ngram_vendi': [float], 'simcse_vendi': [float]}},
                 'responses': {'SelfBLEU': float, 'VENDI': {'bert_vendi': [float], 'semantic_vendi': [float], 'ngram_vendi': [float], 'simcse_vendi': [float]}}
                }
        """
        res = {}

        
        metrics = [self._get_selfbleu_score, self._get_selfjaccard_score, self._get_vendi_and_semantic_score]
        score_names = ["SelfBLEU", "SelfJaccard", "VENDI+SemanticDiversity"]

        for metric, score_names in zip(metrics, score_names):
            split_score_name = score_names.split('+')

            scores = metric()
            for team in ["queries", "responses"]:
                if team not in res:
                    res[team] = {}
                if len(split_score_name) == 1: # the scorer returns one unique score
                    res[team][split_score_name[0]] = scores[team]
                else:  # the scorer returns multiple scores
                    for score, split_name in zip(scores, split_score_name): 
                        res[team][split_name] = score[team]        
        return res

    def _get_selfjaccard_score(self):
        msd = MultisetDistances('None', min_n=self.min_ngram, max_n=self.max_ngram) # the str None is on purpose, as we don't need the reference.
        
        mean_score_dict = {}
        for tokenized_sents, team in zip([self.queries_tokenized, self.responses_tokenized], ["queries", "responses"]):
            mean_score_dict[team] = {}
            if tokenized_sents is None:
                continue

            msd = MultisetDistances(tokenized_sents, min_n=self.min_ngram, max_n=self.max_ngram)
            
            scores = msd.self_jaccard()
            # Normalize the scores between 0 and 1
            scores = np.array(scores) * (len(tokenized_sents)-1) # remove himself/ normalized

            mean_score_dict[team]['batch_mean'] = 1 - np.mean(scores, axis=0)   # 1 - to have a diversity score
            mean_score_dict[team]['batch_std'] = np.std(scores, axis=0)

        return mean_score_dict
        
        
    def _get_selfbleu_score(self):
        """
        Get the SelfBLEU score for both queries and responses TOKENIZED.

        Returns:
            dict: Dictionary containing SelfBLEU scores for both queries and responses.
        """
        selfbleu_scores = {}
        weights = {i: tuple(1. / i for _ in range(i)) for i in range(self.min_ngram, self.max_ngram+1)}
        for tokenized_sents, team in zip([self.queries_tokenized, self.responses_tokenized], ["queries", "responses"]):
            selfbleu_scores[team] = {}
            if tokenized_sents is None:
                continue
            selfbleu = SelfBLEU(tokenized_sents, weights)

            # sum over gram, mean over sentences // but mean over gram for normalizing
            sum_scores = np.asarray(list(selfbleu.get_score().values())).mean(axis=0) # mean over ngram since they are not weighted
            selfbleu_scores[team]['batch_mean'] = 1 - sum_scores.mean(axis=0)   # normalized # 1 - to have a diversity score
            selfbleu_scores[team]['batch_std'] = sum_scores.std(axis=0)

        return selfbleu_scores
    

    def _get_vendi_and_semantic_score(self): # + embedding diversity
        """
        Get the VENDI score for both queries and responses UNTOKENIZED.

        Returns:
            dict: Dictionary containing VENDI scores for both queries and responses.
        """
        vendi_scores = {}
        semantic_scores = {}

        for sents, team in zip([self.queries_untokenized, self.responses_untokenized], ["queries", "responses"]):
            vendi_scores[team] = None
            semantic_scores[team] = None
            if sents is None:
                continue
            team_vendi_scores = {}

            team_vendi_scores['semantic_vendi'], embedding_similarity_matrix, self.vendi_model, self.vendi_tokenizer = text_utils.embedding_vendi_score(sents, model_path="sentence-transformers/all-MiniLM-L6-v2", device=self.device, model=self.vendi_model, tokenizer=self.vendi_tokenizer)

            team_vendi_scores['ngram_vendi'] = text_utils.ngram_vendi_score(sents, ns=list(range(self.min_ngram, self.max_ngram+1)), device=self.device, tokenizer=nltk.word_tokenize)

            # Normalize
            team_vendi_scores['semantic_vendi'] /= len(sents) 
            team_vendi_scores['ngram_vendi'] /= len(sents) 
            vendi_scores[team] = team_vendi_scores

            # # embedding_similarity_matrix is already of normalized embeddings
            np.fill_diagonal(embedding_similarity_matrix,0)

            num_sentences = embedding_similarity_matrix.shape[0]
            scores = np.sum(embedding_similarity_matrix, axis=1)/(num_sentences-1) # remove himself
            emb_scores = {}
            emb_scores['batch_mean'] = 1 - scores.mean() # 1 - to have a diversity score
            emb_scores['batch_std'] = scores.std()
            semantic_scores[team] = emb_scores

        return vendi_scores, semantic_scores
    
    def set_new_queries_responses(self, queries_tokenized: List[List[str]], responses_tokenized: List[List[str]], queries_untokenized: List[str], responses_untokenized: List[str]):
        self.queries_tokenized = queries_tokenized
        self.responses_tokenized = responses_tokenized
        self.queries_untokenized = queries_untokenized
        self.responses_untokenized = responses_untokenized
