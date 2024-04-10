import json
import os

import requests
from backend.shared_configs.model_server_models import IntentRequest, IntentResponse
from danswer.search.enums import EmbedTextType
from danswer.utils.logger import setup_logger
from .base_provider import NLPProvider

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# https://docs.cohere.com/reference/embed
# WARN: embed-multilingual-v3.0 is 1024 dimensions so make sure to set DOC_EMBEDDING_DIM=1024
COHERE_EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

# https://docs.cohere.com/reference/rerank-1
COHERE_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-multilingual-v2.0")

# https://docs.cohere.com/reference/classify
# NOTICE: Suggest using a fine-tuned Cohere model as won't need few shot examples
COHERE_CLASSIFY_MODEL = os.environ.get(
    "COHERE_CLASSIFY_MODEL", "embed-multilingual-v2.0"
)

COHERE_CLASSIFY_EXAMPLES = json.loads(
    os.environ.get(
        "COHERE_CLASSIFY_EXAMPLES",
        '[{"text":"What is the answer to life, the universe, and everything?","label":"Rumination"},{"text":"I need help right now","label":"Urgent"},{"text":"Knock knock...","label":"Joke"}]',
    )
)

logger = setup_logger()


class CohereProvider(NLPProvider):
    # Will vary based on method used
    # TODO: Make this more type safe
    def __init__(self, **kwargs):
        if COHERE_API_KEY is None:
            raise ValueError("COHERE_API_KEY environment variable not found")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def embedder_encode(
        self, texts: list[str], text_type: EmbedTextType, **kwargs
    ) -> list[list[float]]:
        input_type = self._determine_input_type(text_type)
        request_body = {
            "texts": texts,
            "input_type": input_type,
            "model": COHERE_EMBED_MODEL,
        }
        response = make_cohere_request("embed", request_body)
        if response.status_code != 200:
            raise ValueError(f"Failed to embed texts: {response.text}")
        return response.json()["embeddings"]

    def _determine_input_type(self, text_type: EmbedTextType) -> str:
        if text_type == EmbedTextType.PASSAGE:
            return "search_document"
        elif text_type == EmbedTextType.QUERY:
            return "search_query"
        else:
            raise ValueError(f"Unsupported text type: {text_type}")

    def cross_encoder_ensemble_predict(
        self, query: str, passages: list[str]
    ) -> list[list[float]]:
        request_body = {
            "query": query,
            "passages": passages,
            "model": COHERE_RERANK_MODEL,
        }

        response = make_cohere_request("rerank", request_body)

        if response.status_code != 200:
            raise ValueError(f"Failed to rerank passages: {response.text}")

        data = response.json()
        scores = [item["relevance_score"] for item in data["results"]]

        return [[score] for score in scores]

    def intent_predict(self, query: str) -> list[float]:
        # request_body = {
        #     "model": COHERE_CLASSIFY_MODEL,
        #     "inputs": [query],
        # }

        # if len(COHERE_CLASSIFY_EXAMPLES) > 0:
        #     request_body["examples"] = COHERE_CLASSIFY_EXAMPLES

        # response = make_cohere_request("classify", request_body)

        # if response.status_code != 200:
        #     raise ValueError(f"Failed to classify query: {response.text}")

        # data = response.json()

        # classification = data["classifications"][0]

        # TODO: replace below with intent classification approach that uses cohere multilingual model

        intent_request = IntentRequest(query=query)

        response = requests.post(
            self.intent_server_endpoint, json=intent_request.dict()
        )
        response.raise_for_status()

        return IntentResponse(**response.json()).class_probs


def make_cohere_request(path: str, body: dict) -> dict:
    return requests.post(
        f"https://api.cohere.ai/v1/{path}",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {COHERE_API_KEY}",
        },
        json=body,
    )
