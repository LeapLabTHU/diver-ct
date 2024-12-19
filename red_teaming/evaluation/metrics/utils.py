import numpy as np
from typing import Any, Dict, List
import pickle
import os
import warnings
import json


def sample_data(data: List, sample_idx: List[int] = None):
    ''' Sample the data given the sample_idx. If sample_idx is None, return the data.
    '''
    return [data[i] for i in sample_idx] if sample_idx is not None else data


def aggregate_scores(score_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates a list of dictionaries containing scores into a single dictionary containing the list of each scores.
    Args:
    score_dicts (List[Dict[str, Any]]): A list of dictionaries containing scores (float as leaf).
    Returns:
    Dict[str, Any]: A dictionary containing the list of scores for each key (List as leaf)
    """
    def recurse(items, result):
        for key, value in items.items():
            if isinstance(value, dict):  # If the value is a dictionary, recurse further
                if key not in result:
                    result[key] = {}
                recurse(value, result[key])
            elif value is None:  # If the value is None, append None to the list in the result dictionary
                if key in result:
                    result[key].append(None)
                else:
                    result[key] = [None]
            else:
                # It's a leaf, assume it's a float and append to list in the result dictionary
                if key in result:
                    result[key].append(value)
                else:
                    result[key] = [value]
    aggregated_scores = {}
    for score_dict in score_dicts:
        recurse(score_dict, aggregated_scores)
    
    return aggregated_scores

def transform_leaf(node: Any, funcs: Dict[str, callable]={'mean': np.mean,'std': np.std}):
        """
        Transforms each leaf node (list of floats) of a nested dictionary into a dictionary
        containing the mean and std of the list.
        
        Args:
        node (Any): The current node being processed, which can be a dictionary, list, or any other type.
        
        Returns:
        Any: The transformed node.
        """
        # Base case: if the current node is a list of floats, calculate mean and std
        if isinstance(node, list):
            if any(x is None for x in node):
                return None
            elif len(node) == 1:
                return node[0]
            return {name: func(node) for name, func in funcs.items()}
            
        # Recursive case: if the node is a dictionary, apply the function to each of its values
        elif isinstance(node, dict):
            return {key: transform_leaf(value) for key, value in node.items()}

        # If the node is neither a dict nor a list of floats, return it as is (should not happen in this specific use case)
        return node


def save_obs_jsonl(ppo_trainer, red_tokenizer, blue_tokenizer, cur_obs, save_path):
    process_id = os.getpid()
    filename = f'obs_{process_id}.jsonl'
    full_path = os.path.join(save_path, filename)
    cur_obs = cur_obs.to_dict()

    allowed_keys = {'red_prompt', 'blue_response', 'decoded_prompt', 'safety', 'semantic_diversity', 'n_gram_diversity', 'gibberish', 'iteration'}
    cur_obs = {key: (value.tolist() if not isinstance(value, list) else [v.item() for v in value] if not isinstance(value[0], str) else value)
            for key, value in cur_obs.items() if key in allowed_keys and value is not None}

    # Append observation data as a single JSON line to the file
    try:
        with open(full_path, 'a') as f:  # 'a' opens the file for appending
            for i in range(len(cur_obs['red_prompt'])):
                json.dump({k: v[i] for k, v in cur_obs.items()}, f)
                f.write('\n')  # write a newline to separate JSON objects
    except Exception as e:
        print(f"Failed to append data to {full_path}: {e}")

def get_obs_memory(ppo_trainer, red_tokenizer, blue_tokenizer, cur_obs):
    cur_obs = cur_obs.to_dict()
    allowed_keys = {'red_prompt', 'blue_response', 'decoded_prompt', 'safety', 'semantic_diversity', 'n_gram_diversity', 'gibberish', 'iteration'}
    return {key: (value.tolist() if not isinstance(value, list) else [v.item() for v in value] if not isinstance(value[0], str) else value)
            for key, value in cur_obs.items() if key in allowed_keys and value is not None}


def save_obs(ppo_trainer, red_tokenizer, blue_tokenizer, cur_obs, save_path):
            process_id = os.getpid()
            # each process saves its own obs, and concat them.
            filename = f'obs_{process_id}.pkl'
            
            # remove unwanted keys and convert to list
            cur_obs = cur_obs.to_dict()
            
            cur_keys = list(cur_obs.keys()) # Need to copy
            for key in cur_keys:
                if key not in ['red_prompt', 'blue_response', 'decoded_prompt', 'safety', 'semantic_diversity', 'n_gram_diversity', 'gibberish', 'iteration'] or cur_obs[key] is None:
                    cur_obs.pop(key)
                else:
                    if type(cur_obs[key]) == list:
                        if type(cur_obs[key][0]) == str:
                            continue
                        cur_obs[key] = [value.item() for value in cur_obs[key]]
                    else:
                        cur_obs[key] = cur_obs[key].tolist()

            # decode the tokens for red_prompt and blue_response and decoded_prompt
            # for key in ['red_prompt', 'decoded_prompt']:
            #     cur_obs[key] = red_tokenizer.batch_decode(cur_obs[key], skip_special_tokens=True)
            # cur_obs['blue_response'] = blue_tokenizer.batch_decode(cur_obs['blue_response'], skip_special_tokens=True)
            
            if os.path.exists(f'{save_path}/{filename}'):
                with open(f'{save_path}/{filename}', 'rb') as f:
                    merged_dict = pickle.load(f)
                for key, value in cur_obs.items():
                    merged_dict[key] = merged_dict[key] + value
            else:
                merged_dict = cur_obs
                
            # save the dict
            with open(f'{save_path}/{filename}', 'wb') as f:
                pickle.dump(merged_dict, f)

def flatten_scores_dict(y):
    out = {}
    
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '/')
        else:
            out[name[:-1]] = x
    
    flatten(y)
    return out

def convert_obs_to_pkl(observations, pkl_path):
    process_id = os.getpid()
    filename = f'obs_{process_id}.pkl'
    full_path = os.path.join(pkl_path, filename)
    pkl_dict = {}
    for key in observations[1].keys():
        pkl_dict[key] = []
    for obs in observations.values():
        for key, value in obs.items():
            pkl_dict[key].extend(value)
    with open(full_path, 'wb') as pkl_file:
        pickle.dump(pkl_dict, pkl_file)

def convert_jsonl_to_pkl(jsonl_path, delete_jsonl=True):
    """
    Convert a JSONL file to a Pickle file and optionally delete the JSONL file.

    Parameters:
    - jsonl_path (str): The path to the JSONL file.
    - delete_jsonl (bool): Whether to delete the JSONL file after conversion. Default is True.
    """
    # Determine the path for the new Pickle file
    base, ext = os.path.splitext(jsonl_path)
    pkl_path = f"{base}.pkl"

    try:
        import pandas as pd

        data = pd.read_json(jsonl_path, lines=True)

        # Save data to a Pickle file
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump({key: data[key].tolist() for key in data.columns}, pkl_file)
        
        print(f"Data successfully saved to {pkl_path}")

        # Optionally delete the JSONL file
        if delete_jsonl:
            os.remove(jsonl_path)
            print(f"Deleted the original JSONL file: {jsonl_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def merge_pkl(path):
    '''Merge all pkl dict into a single dict.
    Each pkl is a dict.
    The result is a single dict, saved in all.pkl.
    '''

    if os.path.exists(os.path.join(path, 'all.pkl')):
        raise ValueError('all.pkl already exists in the folder. Please do something.')

    all_dict = None
    cur_len = 0
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
                if all_dict is None:
                    all_dict = data
                    cur_len = len(data['red_prompt'])
                else:
                    for key, value in data.items():
                        all_dict[key] = all_dict[key] + value
                        assert len(all_dict['red_prompt']) == cur_len + len(data['red_prompt'])
                    cur_len = len(all_dict['red_prompt'])

    # order the dict by iteration
                    
    if 'iteration' not in all_dict:
        warnings.warn('No iteration key found in logs. Assuming it is already ordered')
        # Sort logs by 'iteration'
    else:
        sort_indices = np.argsort(all_dict['iteration'])
        all_dict = {key: [all_dict[key][idx] for idx in sort_indices] for key in all_dict}

    # all_dict = {key: [x for _, x in sorted(zip(all_dict['iteration'], all_dict[key]))] for key in all_dict.keys()}

    with open(os.path.join(path, 'all.pkl'), 'wb') as f:
        pickle.dump(all_dict, f)
    
    return all_dict
