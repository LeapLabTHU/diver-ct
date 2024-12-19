import itertools

import datasets
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from scipy.sparse import issparse


from red_teaming.evaluation.metrics.vendi_score import vendi, data_utils
from red_teaming.evaluation.metrics.vendi_score.data_utils import Example, Group

# from vendi_score import data_utils, vendi
# from vendi_score.data_utils import Example, Group


def get_tokenizer(model="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    def tokenize(s):
        return tokenizer.convert_ids_to_tokens(tokenizer(s).input_ids)

    return tokenize


def sklearn_tokenizer():
    return CountVectorizer().build_tokenizer()


def get_mnli():
    data = itertools.chain(
        datasets.load_dataset("multi_nli", split="validation_matched"),
        datasets.load_dataset("multi_nli", split="validation_mismatched"),
    )
    seen = set()
    examples = []
    for d in data:
        s = d["premise"]
        if s in seen:
            continue
        seen.add(s)
        examples.append(Example(x=s, labels={"y": d["genre"]}))
    return examples


def get_ngrams(
    sents,
    n=1,
    tokenizer=None,
    return_vectorizer=False,
    lowercase=False,
    **kwargs,
):
    if tokenizer is None:
        tokenizer = word_tokenize
    ngram_range = n if type(n) == tuple else (n, n)
    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        ngram_range=ngram_range,
        lowercase=lowercase,
        **kwargs,
    )
    X = vectorizer.fit_transform(sents)
    if return_vectorizer:
        return X, vectorizer
    return X


def add_ngrams_to_examples(
    examples, n=1, tokenizer=None, return_vectorizer=False, **kwargs
):
    X = get_ngrams([e.x for e in examples], n=n, tokenizer=tokenizer, **kwargs)
    for e, x in zip(examples, X):
        e.features[f"{n}-grams"] = x
    return examples


def get_embeddings(
    sents,
    model=None,
    tokenizer=None,
    batch_size=16,
    device="cpu",
    model_path="princeton-nlp/unsup-simcse-roberta-base",
):
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = AutoModel.from_pretrained(model_path).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    embeddings = []
    for batch in tqdm(data_utils.to_batches(sents, batch_size), desc="Getting embeddings"):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
            if hasattr(output, "pooler_output"):
                output = output.pooler_output
            else:
                output = output.last_hidden_state[:, 0]
        if type(output) == list:
            output = output[0]
        embeddings.append(output.squeeze().cpu().numpy())
    return np.concatenate(embeddings, 0), model, tokenizer


def add_embeddings_to_examples(
    examples,
    model=None,
    tokenizer=None,
    batch_size=16,
    device="cpu",
    model_name="princeton-nlp/unsup-simcse-roberta-base",
    feature_name="unsup_simcse",
):
    X, model, tokenizer = get_embeddings(
        [e.x for e in examples],
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
    )
    for e, x in zip(examples, X):
        e.features[feature_name] = x
    return examples, model, tokenizer


def single_ngram_diversity(sents, n, tokenizer=None, **kwargs):
    X = get_ngrams(sents, n=n, tokenizer=tokenizer, **kwargs)
    distinct = X.shape[-1]
    total = X.sum()
    # unique = (counts == 1).sum()
    # total = counts.shape[-1]
    return distinct / total


def ngram_diversity(sents, ns=[1, 2, 3, 4], tokenizer=None, **kwargs):
    return np.mean(
        [
            single_ngram_diversity(sents, n, tokenizer=tokenizer, **kwargs)
            for n in ns
        ]
    )


def bleu(hyps, refs, tokenizer=None):
    if type(hyps[0]) == str:
        if tokenizer is None:
            tokenizer = word_tokenize
        hyp_tokens = [tokenizer(s) for s in hyps]
        ref_tokens = [tokenizer(s) for s in refs]
    else:
        hyp_tokens = hyps
        ref_tokens = refs
    smoothing = bleu_score.SmoothingFunction().method1
    return np.mean(
        [
            bleu_score.sentence_bleu(refs, hyp, smoothing_function=smoothing)
            for hyp in hyps
        ]
    )


def self_bleu(sents, tokenizer):
    examples = [tokenizer(s) for s in sents]
    smoothing = bleu_score.SmoothingFunction().method1
    scores = []
    for i in range(len(examples)):
        hyp = examples[i]
        ref = examples[:i] + examples[i + 1 :]
        scores.append(
            bleu_score.sentence_bleu(ref, hyp, smoothing_function=smoothing)
        )
    return np.mean(scores)

def semantic_diversity(sents, tokenizer):
    examples = [tokenizer(s) for s in sents]
    smoothing = bleu_score.SmoothingFunction().method1
    scores = []
    for i in range(len(examples)):
        hyp = examples[i]
        ref = examples[:i] + examples[i + 1 :]
        scores.append(
            bleu_score.sentence_bleu(ref, hyp, smoothing_function=smoothing)
        )
    return np.mean(scores)

def pairwise_bleu(sents, tokenizer):
    examples = [tokenizer(s) for s in sents]
    smoothing = bleu_score.SmoothingFunction().method1
    scores = []
    for i in range(len(examples)):
        lst = []
        for j in range(len(examples)):
            if j == i:
                continue
            hyp = examples[i]
            ref = [examples[j]]
            lst.append(
                bleu_score.sentence_bleu(ref, hyp, smoothing_function=smoothing)
            )
        scores.append(np.mean(lst))
    return np.mean(scores)


def ngram_vendi_score_notworking(sents, ns=[1, 2, 3, 4], tokenizer=None, device='cpu', **kwargs):
    
    final_K = None
    num_chunks = 10  # Define the number of chunks

    pbar = tqdm(len(ns)*num_chunks, desc="Computing ngram for vendi score...")
    for n in ns:
        X = get_ngrams(sents, n=n, tokenizer=tokenizer)
        
        if issparse(X):
            X = X.toarray()  # Convert sparse matrix to dense

        # Convert to tensor, still on CPU
        X = torch.tensor(X, dtype=torch.float32)
        
        # Normalize and compute K = X @ X^T on CPU
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        # remove X from memory
        K = torch.matmul(X, X.T)
        if final_K is None:
            final_K = K
            pbar.update(num_chunks)
            
        else:
            chunk_size = int(np.ceil(final_K.shape[0] / num_chunks))

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, final_K.shape[0])
                # sum on gpu and store in final_K

                K_chunk = K[start_idx:end_idx].to(device) + final_K[start_idx:end_idx].to(device)
                if n == ns[-1]:
                    K_chunk /= len(ns)
                final_K[start_idx:end_idx] = K_chunk.to('cpu')

                pbar.update(1)
        
    # Convert the combined final mean matrix back to numpy for vendi scoring
    final_K = final_K.numpy()

    return vendi.score_K(final_K)

def ngram_vendi_score_oom(sents, ns=[1, 2, 3, 4], tokenizer=None, device='cpu', **kwargs):
    Ks = []
    for n in ns:
        X = normalize(get_ngrams(sents, n=n, tokenizer=tokenizer))
        Ks.append((X @ X.T).A)
    # K = np.stack(Ks, axis=0).mean(axis=0)
    Ks_tensors = [torch.tensor(K, device=device) for K in Ks]
    Ks_stacked = torch.stack(Ks_tensors, dim=0)
    # Compute the mean of the stacked tensors on the GPU
    K_mean = Ks_stacked.mean(dim=0)
    # Move the mean tensor back to CPU and convert to a NumPy
    K_mean_np = K_mean.cpu().numpy()
    return vendi.score_K(K_mean_np)

def ngram_vendi_score(sents, ns=[1, 2, 3, 4], tokenizer=None, **kwargs):
    Ks = []
    for n in ns:
        X = normalize(get_ngrams(sents, n=n, tokenizer=tokenizer))
        Ks.append((X @ X.T).A)
    # K = np.stack(Ks, axis=0).mean(axis=0)
    K = np.stack(Ks, axis=0)
    K = K.mean(axis=0)
    return vendi.score_K(K)


def embedding_vendi_score( # Modified to return the model and tokenizer and dot product
    sents,
    model=None,
    tokenizer=None,
    batch_size=16,
    device="cpu",
    model_path="princeton-nlp/unsup-simcse-roberta-base",
):
    X, model, tokenizer = get_embeddings(
        sents,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        model_path=model_path,
    )
    n, d = X.shape
    if n < d:
        vendi_score, embed_matrix = vendi.score_X(X)
        return vendi_score, embed_matrix, model, tokenizer   #0 is score, 1 is emb_matrix, score_X = score, X @ X.T
    vendi_score, embed_matrix = vendi.score_dual(X)
    return vendi_score, embed_matrix, model, tokenizer  # score_dual = score, X.T @ X
