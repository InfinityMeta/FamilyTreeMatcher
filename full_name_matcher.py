import os
import pickle
import string
from collections import Counter
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd
from nltk.util import ngrams
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ALPHABET_LETTERS = list(string.ascii_lowercase)
POSSIBLE_BIGRAMS = [l1 + l2 for l1, l2 in product(ALPHABET_LETTERS, ALPHABET_LETTERS)]


class FullNameMatcher:
    """Class for matching people fullnames.

    Attributes:
        distance_metric: Metric to use for training nearest neigbors model.
    """

    def __init__(self, distance_metric: str = "cosine") -> None:
        """Initializes FullNameMatcher class object.

        Args:
            distance_metric: metric to use for training nearest neigbors model
        """
        self.distance_metric = distance_metric

    def fit(self, entities_paths: Dict[str, str]) -> None:
        """Fits full name matcher model.

        Args:
            entities_paths: Mapping entity -> path to entity`s values.
                For example:
                entities_paths = {
                    'name': './data/russian_trans_names.csv',
                    'surname': './data/russian_trans_surnames.csv'
                }
        """
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./entities", exist_ok=True)

        for entity, path in tqdm(entities_paths.items()):
            entity_df = pd.read_csv(path)
            entity_df[entity].to_csv(f"./entities/{entity}.csv", index=False)

            bigrams_vectors = self.prepare_bigrams_vectors(entity, entity_df)
            neigh = NearestNeighbors(metric=self.distance_metric)
            neigh.fit(bigrams_vectors)

            with open(f"./models/{entity}_nearest_neighbors.pkl", "wb") as f:
                pickle.dump(neigh, f)

    def search(
        self, entities_dict: Dict[str, str], neighbors_num: int = 1
    ) -> Dict[str, List[str]]:
        """Searches for the most similar names and surnames for received pair (name, surname).

        Args:
            entities_dict: Mapping entity -> entity value.
                For example:
                entities_dict = {
                    'name': 'aleksej',
                    'surname': 'romanov'
                }
            neighbors_num: Number of similar values to search for each entity.

        Returns:
            Mapping entity -> most similar entity`s values.
        """
        search_result_dict = {}
        for entity, value in entities_dict.items():
            value = pd.json_normalize(
                Counter(list([l1 + l2 for l1, l2 in ngrams(value, 2)]))
            )
            available_bigrams = set(POSSIBLE_BIGRAMS).intersection(set(value.columns))
            not_included_bigrams = set(POSSIBLE_BIGRAMS) - set(available_bigrams)
            not_included_bigrams_df = pd.DataFrame(
                np.zeros((1, len(not_included_bigrams))),
                columns=list(not_included_bigrams),
            )
            bigrams_raw = pd.concat([value, not_included_bigrams_df], axis=1)

            with open(f"./models/{entity}_standard_scaler.pkl", "rb") as f:
                sc = pickle.load(f)

            bigram_vector = sc.transform(bigrams_raw[[*POSSIBLE_BIGRAMS]].to_numpy())

            with open(f"./models/{entity}_nearest_neighbors.pkl", "rb") as f:
                neigh = pickle.load(f)

            entity_values = pd.read_csv(f"./entities/{entity}.csv")
            nearest_idxs = neigh.kneighbors(
                bigram_vector, neighbors_num, return_distance=False
            )
            entity_obtained_values = [
                entity_values.loc[idx].values[0] for idx in nearest_idxs[0]
            ]
            search_result_dict[entity] = entity_obtained_values

        return search_result_dict

    @staticmethod
    def prepare_bigrams_vectors(entity: str, entity_df: pd.DataFrame) -> np.array:
        """Prepares bigrams vectors based on entity`s DataFrame.

        Args:
            entity: Name of entity.
            entity_df: DataFrame with entity`s values.

        Returns:
            Array with bigrams vectors.
        """
        entity_df[f"decomposed_{entity}"] = entity_df[entity].apply(
            lambda x: Counter(list([l1 + l2 for l1, l2 in ngrams(x, 2)]))
        )
        entity_df_decomposed = pd.json_normalize(entity_df[f"decomposed_{entity}"])
        available_bigrams = set(POSSIBLE_BIGRAMS).intersection(
            set(entity_df_decomposed.columns)
        )
        not_included_bigrams = set(POSSIBLE_BIGRAMS) - set(available_bigrams)
        not_included_bigrams_df = pd.DataFrame(
            np.zeros((len(entity_df_decomposed), len(not_included_bigrams))),
            columns=list(not_included_bigrams),
        )
        entity_df = pd.concat([entity_df_decomposed, not_included_bigrams_df], axis=1)
        entity_df = entity_df[[*POSSIBLE_BIGRAMS]]
        entity_df = entity_df.fillna(0.0)

        sc = StandardScaler()
        bigrams_vectors = sc.fit_transform(entity_df.values)

        with open(f"./models/{entity}_standard_scaler.pkl", "wb") as f:
            pickle.dump(sc, f)

        return bigrams_vectors
