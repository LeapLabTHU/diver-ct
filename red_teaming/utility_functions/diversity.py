from typing import List, Union
from collections import deque

import nltk
from fast_bleu import BLEU
import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from trl.trainer.utils import disable_dropout_in_model

from red_teaming.utility_functions.embedders import get_embedder
from red_teaming.utility_functions.metrics import Metrics


class SelfBLEUScore(Metrics):
    """
    This class is used to calculate the novelty reward for a given sentence.
    Input:
        K: int, the maximum gram to be considered, summing from 1 to K
        sample_size: int, the number of sentences to be sampled from the references
    
    Usage:
        1. Initialize the class
        2. Call get_scores() and pass the hypothesis sentences (not tokenized) 
        3. The input hypothesis sentences will be tokenized and compared with the references
        4. The tokenized hypothesis sentences will be added to the self.references
    """
    def __init__(self, K: int = 5, sample_size: int = 100000):
        super().__init__()
        self.name = 'BLEU Diversity'
        self.K = K
        assert K >= 2, "K must be greater than 1"
        self.weights = {i: tuple(1. / i for _ in range(i)) for i in range(2, K+1)}
        self.references = []                             # tokenized references
        self._sample_size = sample_size

    def update_references(self, new_references: List[str], tokenize: bool = True):
        """
        Add the new references for the novelty reward
        Input:
            new_references: list of strings, each string is a reference sentence to be added.
        """
        if tokenize:
            new_references = [nltk.word_tokenize(ref) for ref in new_references]
        self.references.extend(new_references)
        if self._sample_size < 0 or len(self.references) <= self._sample_size:
            references = self.references
        else:
            references = np.random.choice(len(self.references), self._sample_size, replace=False)
        self.bleu = BLEU(references, self.weights)

    def get_reference(self):
        return self.references

    def get_score(self, hypothesis: List[str], append: bool = False, tokenize: bool = True, **kwargs) -> torch.Tensor:
        # initialize the references if it is empty
        if len(self.references) == 0:
            return torch.zeros(len(hypothesis))
        
        tokenized_hypothesis = [nltk.word_tokenize(hypo) for hypo in hypothesis] if tokenize else hypothesis

        # rewards = (1 - torch.tensor([s for s in self.bleu.get_score(tokenized_hypothesis).values()]).sum(0))
        rewards = (- torch.tensor([s for s in self.bleu.get_score(tokenized_hypothesis).values()]).mean(0))

        if append:
            self.update_references(tokenized_hypothesis, tokenize=False)

        return rewards


class SemanticDiversityScore(Metrics):
    """
    This class is used to calculate the semantic diversity reward for a given sentence. Using KNN
    Input:
        embedder_model_name: str, the name of the embedder model to be used
        sample_size: int, the number of sentences to be sampled from the references
        
    Usage:
        1. Initialize the class
        2. Call get_scores() and pass the hypothesis sentences (not tokenized)
        3. The input hypothesis sentences will be embedded and compared with the references
        4. The embedded hypothesis sentences will be added to the self.references
    """
    def __init__(
        self,
        embedder_model_name: str,
        top_k: Union[int, str],
        distance_strategy: str = 'cosine',
        **kwargs,
    ):
        super().__init__()
        self.name = 'Semantic Diversity'
        self.embedder = get_embedder(embedder_model_name, True)
        self.db = None
        self.model = self.embedder.module
        self.top_k = top_k
        if distance_strategy == 'euclidean':
            self.distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
            self.normalize_L2 = False
            # from the APT paper, reward: log(1 + mean(scores))
            self.post_processor = lambda similarity: np.log(1 + similarity)
        elif distance_strategy == 'cosine':
            self.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
            self.normalize_L2 = True
            self.post_processor = lambda similarity: - similarity
        elif distance_strategy == 'inner_product':
            self.distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
            self.normalize_L2 = False
            self.post_processor = lambda similarity: 1 - np.log(similarity)
        else:
            raise NotImplementedError
        print(self.model)

    def update_references(self, new_references: List[str], **kwargs):
        if self.db is None:
            self.db = FAISS.from_texts(
                new_references,
                self.embedder,
                distance_strategy=self.distance_strategy,
                normalize_L2=self.normalize_L2,
            )
        else:
            self.db.add_texts(new_references)

    def get_score(self, hypothesis: List[str], append = False, **kwargs) -> torch.Tensor:
        # first time querying, need to init the vdb
        if self.db is None:
            return torch.zeros(len(hypothesis))

        bcos_score = []
        if self.top_k == 'all':
            top_k = self.db.index.ntotal
        else:
            top_k = int(self.top_k)

        for hyp in hypothesis:
            similarities = [s for _, s in self.db.similarity_search_with_score(hyp, k=top_k)]
            bcos_score.append(self.post_processor(np.mean(similarities)))

        if append:
            self.update_references(hypothesis)

        return torch.tensor(bcos_score)


class SampledSemanticDiversityScore(Metrics):
    """
    Sampled version of SemanticDiversityScore
    """
    def __init__(self, embedder_model_name: str, sample_size: int = 100000, **kwargs,): 
        super().__init__()
        self.name = 'Semantic Diversity'
        self.embedder = get_embedder(embedder_model_name, False)
        self.references = None                      # embedded references normalized
        self.sample_size = sample_size              # number of references to be sampled
        self.model = self.embedder.module
        print(self.model)

    def update_references(self, new_references: List[str], embed: bool = True):
        if embed:
            new_references = self.embedder.embed_documents(new_references)
            new_references = F.normalize(new_references, p=2, dim=1)
        if self.references is None:
            self.references = new_references
        else: 
            self.references = torch.cat((self.references, new_references), dim=0)

    def get_score(self, hypothesis: List[str], append = False, **kwargs) -> torch.Tensor:
        embedded_hypothesis = self.embedder.embed_query(hypothesis)
        # normalize the embedded hypothesis
        embedded_hypothesis_norm = F.normalize(embedded_hypothesis, p=2, dim=1)

        if self.references is None:
            return torch.zeros(len(hypothesis))
        bcos_score = []

        if self.sample_size < 0 or len(self.references) <= self.sample_size:
            references = self.references
        else:
            # randomly sample from the references
            references = self.references[np.random.choice(len(self.references), self.sample_size, replace=False)]

        # matrix multiplication
        bcos_score = pairwise_cosine_similarity(embedded_hypothesis_norm, references)

        # subtract the mean of the bcos_score, since more reward is given to more diverse sentences
        bcos_score = -bcos_score.mean(1)

        if append:
            self.update_references(embedded_hypothesis_norm, embed=False)

        return bcos_score
