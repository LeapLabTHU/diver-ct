from datasets import load_dataset, concatenate_datasets, DatasetDict
import nltk
from typing import List

import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
red_teaming_directory = current_directory.split('red_teaming')[0] + 'red_teaming'

sys.path.insert(0, red_teaming_directory)

from red_teaming.evaluation.metrics.multiset_distance import MultisetDistances
from red_teaming.evaluation.metrics.vendi_score.text_utils import get_embeddings
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer
import numpy as np


class DistributionScores:
    def __init__(self, queries_tokenized: List[List[str]], responses_tokenized: List[List[str]],
                 min_ngram: int = 2, max_ngram: int = 5, device: str = 'cpu'):
        """
        Initialize DistributionScores with specified n-gram range and tokenized queries and responses.

        Args:
            min_ngram (int): Minimum size of n-gram for multiset distances.
            max_ngram (int): Maximum size of n-gram for multiset distances.
            queries_tokenized (List[List[str]]): List of tokenized queries. Can be None
            responses_tokenized (List[List[str]]): List of tokenized responses. Can be None
        """
        self.queries_tokenized = queries_tokenized
        self.responses_tokenized = responses_tokenized
        
        self.embedded_queries = None
        self.embedded_responses = None
        
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        # self.distribution_metrics = ['jaccard', 'sorensen', 'canberra', 'BLEU']
        self.distribution_metrics = ['jaccard', 'BLEU', 'cossemb']

        self.loaded_datasets = {} # Cache for loaded datasets
        self.device = device
        
        # for embedding similarity
        self.embedder_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_fast=True)
        self.datasets_embed_matrix = {} 

    
    def get_score_datasets(self, dataset_name: List[str], num_samples: int = -1, split: str = 'test'):
        """
        Computes distribution distance between model outputs and reference datasets.

        Args:
            dataset_name (List[str]): List of reference dataset names or paths.

        Returns:
            dict: Dictionary containing distribution scores for each metric, team, and dataset.
        """
        # Initialize to None if queries/responses are not provided
        query_msd = MultisetDistances(self.queries_tokenized, min_n=self.min_ngram, max_n=self.max_ngram) if self.queries_tokenized else None
        response_msd = MultisetDistances(self.responses_tokenized, min_n=self.min_ngram, max_n=self.max_ngram) if self.responses_tokenized else None 
        
        mean_score_dict = {"queries": {}, "responses": {}}
        for dataset in dataset_name:
            # check if the dataset is already loaded
            if dataset in self.loaded_datasets:
                ref_red_query, ref_blue_response = self.loaded_datasets[dataset]
                ref_red_emb, ref_blue_emb = self.datasets_embed_matrix[dataset]["queries"], self.datasets_embed_matrix[dataset]["responses"]
            else: # not cached, need to load
                ref_red_query, ref_blue_response = self._load_dataset(dataset, num_samples = num_samples, split = split)
                
                # embed the queries/responses of dataset
                if ref_red_query:
                    ref_red_emb, _, _ = get_embeddings(sents=ref_red_query, model=self.embedder_model, tokenizer=self.tokenizer, device=self.device, model_path='sentence-transformers/all-MiniLM-L6-v2')
                    ref_red_emb = preprocessing.normalize(ref_red_emb, axis=1)
                else:
                    ref_red_emb = None
                
                if ref_blue_response:
                    ref_blue_emb, _, _ = get_embeddings(sents=ref_blue_response, model=self.embedder_model, tokenizer=self.tokenizer, device=self.device, model_path='sentence-transformers/all-MiniLM-L6-v2')
                    ref_blue_emb = preprocessing.normalize(ref_blue_emb, axis=1)
                else:
                    ref_blue_emb = None
                # cache them
                self.datasets_embed_matrix[dataset] = {"queries": ref_red_emb, "responses": ref_blue_emb}
                
            for reference, msd, team in zip([ref_red_query, ref_blue_response], [query_msd, response_msd], ["queries", "responses"]):
                # If the dataset does not have queries/responses OR generated queries/responses are not provided, skip it
                if reference is None or msd is None: 
                    mean_score_dict[team][dataset] = None
                else:
                    mean_score_dict[team][dataset] = {}
                    # Tokenize the references if not None
                    reference = [nltk.word_tokenize(x) for x in reference]
                    for metric_name in self.distribution_metrics:
                        if metric_name == 'cossemb':
                            embedded_team = getattr(self, f'embedded_{team}')
                            sim_matrix = embedded_team @ self.datasets_embed_matrix[dataset][team].T
                            sim_score = np.mean(np.mean(sim_matrix, axis=1), axis=0)
                            mean_score_dict[team][dataset][metric_name] = sim_score
                            continue
                        # print(f'Calculating {metric_name} score for {team} in {dataset}...')
                        get_metric_score = getattr(msd, f'get_{metric_name}_score')
                        mean_score_dict[team][dataset][metric_name] = get_metric_score(reference)
        return mean_score_dict
    
    def get_score_raw_data(self, tokenized_queries_responses: dict):
            """
            Computes distribution distance between model outputs and another model outputs.
            
            Args:
                raw_data (dict): Dictionary containing TOKENIZED "queries" and "responses". One or the other can be None.

            Returns:
                dict: Dictionary containing distribution scores for each metric, team, and dataset.
            """
            # Initialize to None if queries/responses are not provided
            query_msd = MultisetDistances(self.queries_tokenized, min_n=self.min_ngram, max_n=self.max_ngram) if self.queries_tokenized else None
            response_msd = MultisetDistances(self.responses_tokenized, min_n=self.min_ngram, max_n=self.max_ngram) if self.responses_tokenized else None
            
            mean_score_dict = {"queries": {}, "responses": {}}

            for team, sents, msd in zip(["queries", "responses"], [tokenized_queries_responses["queries"], tokenized_queries_responses["responses"]],[query_msd, response_msd]):

                # If the dataset does not have queries/responses, skip it
                if sents is None or msd is None:
                    mean_score_dict[team] = None
                else:
                    mean_score_dict[team] = {}
                    # Tokenize the references if not None
                    for metric_name in self.distribution_metrics:
                        # print(f'Calculating {metric_name} score for {team}...')
                        get_metric_score = getattr(msd, f'get_{metric_name}_score')
                        mean_score_dict[team][metric_name] = get_metric_score(sents)
            return mean_score_dict        
    
    def _load_dataset(self, dataset_path_or_name: str, split: str = 'train', num_samples: int = -1):
        # If already loaded, return the cached dataset
        if dataset_path_or_name in self.loaded_datasets:
            return self.loaded_datasets[dataset_path_or_name]
        
        
        if dataset_path_or_name == "PKU-Alignment/PKU-SafeRLHF":
            ref_dataset = load_dataset(dataset_path_or_name, cache_dir='./datatest/', download_mode='reuse_cache_if_exists', trust_remote_code=True) 

            ref_dataset = ref_dataset[split]
            ref_dataset = ref_dataset.filter(lambda x: x['is_response_0_safe'] == 0 and x['is_response_1_safe'] == 0)
            ref_dataset = [[x['prompt'], x[f"response_{x['better_response_id']}"]] for x in ref_dataset]
            
            # we only take a subset of 10 000 samples
            ref_dataset = ref_dataset[:10000]
            # ref_dataset['red_prompt'] = ref_dataset.pop('prompt')
            # ref_dataset['blue_response'] = ref_dataset.pop('response_1')
            self.loaded_datasets[dataset_path_or_name] = ([x[0] for x in ref_dataset], [x[1] for x in ref_dataset])

            # self.loaded_datasets[dataset_path_or_name] = (ref_dataset['red_prompt'], ref_dataset['blue_response'])
            return [x[0] for x in ref_dataset], [x[1] for x in ref_dataset]
        
        elif dataset_path_or_name == "skg/toxigen-data":
            ref_dataset = load_dataset(dataset_path_or_name, cache_dir='./datatest/', download_mode='reuse_cache_if_exists', trust_remote_code=True) 
            
            
            # For toxigen we merge both train and test splits to get more data. 
            dataset_dict = DatasetDict({"dataset1": ref_dataset['train'], "dataset2": ref_dataset['test']})
            ref_dataset = concatenate_datasets([dataset_dict[key] for key in dataset_dict])
            
            ref_dataset = ref_dataset.filter(lambda x: x['toxicity_human'] >= 3)
            
            self.loaded_datasets[dataset_path_or_name] = (None, ref_dataset['text'])
            return None, ref_dataset['text']
        else:
            raise ValueError(f'Dataset not implemented: {dataset_path_or_name}')

    def set_new_queries_responses(self, queries_tokenized: List[List[str]], responses_tokenized: List[List[str]], 
                                    queries_untokenized: List[str], responses_untokenized: List[str]):
        self.queries_tokenized = queries_tokenized
        self.responses_tokenized = responses_tokenized
        
        self.embedded_queries, _, _ = get_embeddings(sents=queries_untokenized, model=self.embedder_model, tokenizer=self.tokenizer, device=self.device, model_path='sentence-transformers/all-MiniLM-L6-v2')
        self.embedded_queries = preprocessing.normalize(self.embedded_queries, axis=1)
        self.embedded_responses, _, _ = get_embeddings(sents=responses_untokenized, model=self.embedder_model, tokenizer=self.tokenizer, device=self.device, model_path='sentence-transformers/all-MiniLM-L6-v2')
        self.embedded_responses = preprocessing.normalize(self.embedded_responses, axis=1)
