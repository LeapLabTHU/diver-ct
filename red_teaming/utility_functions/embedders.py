from typing import List, Callable

import numpy as np
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class Embedder(Embeddings):
    def __init__(self, model_name: str, to_numpy: bool) -> None:
        self.model_name = model_name
        if model_name == 'bge':
            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'cuda'}
            encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
            self.model =  HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
            self.module = self.model.client
        elif model_name == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.module = self.model
        else:
            raise NotImplementedError
        self.to_numpy = to_numpy

    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.model_name == 'bge':
            embedding = np.array(self.model.embed_documents(texts))
        elif self.model_name == 'uae':
            embedding = self.model.encode([{'text': text} for text in texts], to_numpy=True)
        elif self.model_name == 'minilm':
            embedding = self.model.encode(texts, convert_to_tensor=True, convert_to_numpy=False,show_progress_bar=False)
        if self.to_numpy:
            embedding = embedding.cpu().numpy()
        return embedding

    @torch.no_grad()
    def embed_query(self, query: str) -> np.ndarray:
        if self.model_name == 'bge':
            embedding = np.array(self.model.embed_query(query))
        elif self.model_name == 'uae':
            embedding = self.model.encode({'text': query}, to_numpy=True)
            if embedding.ndim == 2:
                embedding = embedding[0, :]
        elif self.model_name == 'minilm':
            embedding = self.model.encode(query, convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=False)
        if self.to_numpy:
            embedding = embedding.cpu().numpy()
        return embedding


def get_embedder(embedder: str, to_numpy: bool) -> Callable:
    return Embedder(embedder, to_numpy)
