import requests
from danswer.search.enums import EmbedTextType
from danswer.utils.logger import setup_logger
from shared_configs.model_server_models import (
    EmbedRequest,
    EmbedResponse,
    IntentRequest,
    IntentResponse,
    RerankRequest,
    RerankResponse,
)
from .base_provider import NLPProvider

logger = setup_logger()


class LocalModelServerProvider(NLPProvider):
    # Will vary based on method used
    # TODO: Make this more type safe
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def embedder_encode(
        self, texts: list[str], text_type: EmbedTextType, **kwargs
    ) -> list[list[float]]:
        logger.info(f"DEBUG: Embedding {len(texts)} texts")
        if text_type == EmbedTextType.QUERY and self.query_prefix:
            prefixed_texts = [self.query_prefix + text for text in texts]
        elif text_type == EmbedTextType.PASSAGE and self.passage_prefix:
            prefixed_texts = [self.passage_prefix + text for text in texts]
        else:
            prefixed_texts = texts

        embed_request = EmbedRequest(
            texts=prefixed_texts,
            model_name=self.model_name,
            max_context_length=self.max_seq_length,
            normalize_embeddings=self.normalize,
        )

        response = requests.post(self.embed_server_endpoint, json=embed_request.dict())
        response.raise_for_status()

        return EmbedResponse(**response.json()).embeddings

    def cross_encoder_ensemble_predict(
        self, query: str, passages: list[str]
    ) -> list[list[float]]:
        rerank_request = RerankRequest(query=query, documents=passages)

        response = requests.post(
            self.rerank_server_endpoint, json=rerank_request.dict()
        )
        response.raise_for_status()

        return RerankResponse(**response.json()).scores

    def intent_predict(self, query: str) -> list[float]:
        intent_request = IntentRequest(query=query)

        response = requests.post(
            self.intent_server_endpoint, json=intent_request.dict()
        )
        response.raise_for_status()

        return IntentResponse(**response.json()).class_probs
