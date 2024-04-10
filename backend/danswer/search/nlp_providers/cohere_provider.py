from danswer.search.enums import EmbedTextType
from shared_configs.model_server_models import (
    EmbedRequest,
    EmbedResponse,
    IntentRequest,
    IntentResponse,
    RerankRequest,
    RerankResponse,
)
from .base_provider import NLPProvider


class CohereProvider(NLPProvider):
    # Will vary based on method used
    # TODO: Make this more type safe
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def embedder_encode(
        self, texts: list[str], text_type: EmbedTextType, **kwargs
    ) -> list[list[float]]:
        pass

    def cross_encoder_ensemble_predict(
        self, query: str, passages: list[str]
    ) -> list[list[float]]:
        pass

    def intent_predict(self, query: str) -> list[float]:
        pass
