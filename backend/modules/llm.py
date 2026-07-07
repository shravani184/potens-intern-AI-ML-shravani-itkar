from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, List

from backend.core.exceptions import LLMError
from backend.core.logging_config import get_logger
from config import settings

logger = get_logger(__name__)


ANSWER_SYSTEM_PROMPT = (
    "You are a meticulous retrieval-augmented assistant. You answer questions "
    "STRICTLY and ONLY using the provided context passages. Follow these rules "
    "without exception:\n"
    "1. Use only information contained in the CONTEXT. Never rely on prior "
    "knowledge or make assumptions.\n"
    "2. If the context does not contain enough information to answer, you MUST "
    "set \"insufficient_context\" to true and return the exact fallback answer.\n"
    "3. Cite the passages you actually used by their bracket numbers.\n"
    "4. Do not invent citations, facts, figures, or names.\n"
    "5. Respond with a SINGLE valid JSON object and nothing else."
)

ANSWER_USER_TEMPLATE = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "{{\n"
    '  "answer": "<answer grounded only in the context, or the exact fallback sentence>",\n'
    '  "used_passages": [<integers of the passages you used, e.g. 1, 3>],\n'
    '  "insufficient_context": <true|false>\n'
    "}}\n\n"
    'If the context is insufficient, set "insufficient_context" to true and set '
    '"answer" to exactly: "{fallback}"'
)

CONTRADICT_SYSTEM_PROMPT = (
    "You are a rigorous fact-comparison assistant. You are given evidence "
    "passages from two different documents about a specific topic. Determine "
    "whether the two documents CONTRADICT each other on that topic. Base your "
    "judgement ONLY on the supplied passages. A contradiction means the "
    "documents make claims that cannot both be true. Differences in wording, "
    "scope, or detail are NOT contradictions. Respond with a SINGLE valid JSON "
    "object and nothing else."
)

CONTRADICT_USER_TEMPLATE = (
    "TOPIC: {topic}\n\n"
    "=== DOCUMENT 1 ({doc1}) PASSAGES ===\n{context1}\n\n"
    "=== DOCUMENT 2 ({doc2}) PASSAGES ===\n{context2}\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "{{\n"
    '  "conflict": <true|false>,\n'
    '  "reasoning": "<concise explanation grounded in the passages>",\n'
    '  "document_1_passages": [<passage numbers from document 1 you relied on>],\n'
    '  "document_2_passages": [<passage numbers from document 2 you relied on>]\n'
    "}}"
)


class LLMClient:
    """Provider-agnostic chat client built on LangChain chat models."""

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def _build_llm(self):
        if self.provider == "groq":
            if not settings.groq_api_key:
                raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:  # pragma: no cover
                raise LLMError("langchain-groq is not installed.") from exc
            logger.info("Initialising Groq model '%s'.", settings.groq_model)
            return ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        if self.provider == "gemini":
            if not settings.google_api_key:
                raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("langchain-google-genai is not installed.") from exc
            logger.info("Initialising Gemini model '%s'.", settings.gemini_model)
            return ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=settings.gemini_model,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
            )

        raise LLMError(f"Unknown LLM provider: {self.provider}")

    def _chat(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self.llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM invocation failed: {exc}") from exc
        return response.content if hasattr(response, "content") else str(response)

    def generate_answer(self, question: str, context: str) -> Dict:
        user = ANSWER_USER_TEMPLATE.format(
            context=context,
            question=question,
            fallback=settings.insufficient_context_message,
        )
        raw = self._chat(ANSWER_SYSTEM_PROMPT, user)
        logger.info("Raw LLM answer: %s", raw)
        return _parse_json(
            raw,
            default={
                "answer": settings.insufficient_context_message,
                "used_passages": [],
                "insufficient_context": True,
            },
        )

    def detect_contradiction(
        self, topic: str, doc1: str, doc2: str, context1: str, context2: str
    ) -> Dict:
        user = CONTRADICT_USER_TEMPLATE.format(
            topic=topic,
            doc1=doc1,
            doc2=doc2,
            context1=context1,
            context2=context2,
        )
        raw = self._chat(CONTRADICT_SYSTEM_PROMPT, user)
        return _parse_json(
            raw,
            default={
                "conflict": False,
                "reasoning": "Unable to determine a contradiction from the evidence.",
                "document_1_passages": [],
                "document_2_passages": [],
            },
        )


def _parse_json(raw: str, default: Dict) -> Dict:
    if not raw:
        return default
    text = raw.strip()
    
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse LLM JSON output; using default.")
    return default


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the cached LLM-client singleton."""
    return LLMClient()
