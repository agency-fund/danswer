from abc import ABC, abstractmethod
from typing import List
from danswer.search.enums import EmbedTextType


class NLPProvider(ABC):
    @abstractmethod
    def embedder_encode(
        self, texts: List[str], text_type: EmbedTextType
    ) -> List[List[float]]:
        """
        Encodes a list of texts into embeddings.

        :param texts: A list of texts to encode.
        :param text_type: The type of text (e.g., query or passage).
        :return: A list of embeddings.
        """
        pass

    @abstractmethod
    def cross_encoder_ensemble_predict(
        self, query: str, passages: List[str]
    ) -> List[List[float]]:
        """
        Uses a cross-encoder to predict the relevance scores between a query and a list of passages.

        :param query: The query text.
        :param passages: A list of passages to score against the query.
        :return: A list of relevance scores.
        """
        pass

    @abstractmethod
    def intent_predict(self, query: str) -> List[float]:
        """
        Predicts the intent of a query.

        :param query: The query text.
        :return: A list of intent probabilities.
        """
        pass
