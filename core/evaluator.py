"""Offline retrieval metrics and explicit, auditable grounding checks.

No model calls, downloads, or optional imports occur at module import time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import unicodedata
from typing import Iterable, Protocol, Sequence


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def tokens(text: str) -> tuple[str, ...]:
    """NFC Unicode word tokens; punctuation separates words, accents remain.

    This is a reference tokenizer, not a language-specific segmenter.
    """
    _text(text, "text")
    normalized = unicodedata.normalize("NFC", text).casefold()
    words = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                    else " " for c in normalized)
    return tuple(words.split())


@dataclass(frozen=True)
class Document:
    id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.id, "document id")
        _text(self.text, "document text")


def _documents(documents: Iterable[Document]) -> tuple[Document, ...]:
    result = tuple(documents)
    if not result or any(not isinstance(d, Document) for d in result):
        raise ValueError("documents must contain Document instances")
    if len({d.id for d in result}) != len(result):
        raise ValueError("document ids must be unique")
    return result


def _k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


class Retriever(Protocol):
    def search(self, query: str, k: int) -> Sequence[str]:
        """Return unique document ids in descending relevance order."""
        ...


class LexicalRetriever:
    """Small, deterministic bag-of-words cosine baseline; no learned semantics."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = _documents(documents)
        self._vectors = tuple(Counter(tokens(d.text)) for d in self.documents)

    def search(self, query: str, k: int = 5) -> tuple[str, ...]:
        _k(k)
        vector = Counter(tokens(query))
        if not vector:
            raise ValueError("query must contain searchable characters")
        norm = math.sqrt(sum(n * n for n in vector.values()))
        ranked = []
        for doc, candidate in zip(self.documents, self._vectors):
            denominator = norm * math.sqrt(sum(n * n for n in candidate.values()))
            score = sum(n * candidate[t] for t, n in vector.items()) / denominator if denominator else 0.0
            if score > 0:
                ranked.append((doc.id, score))
        return tuple(doc_id for doc_id, _ in sorted(ranked, key=lambda row: (-row[1], row[0]))[:k])


class SentenceTransformerRetriever:
    """Optional multilingual cosine retrieval. Initialization may download a model."""

    def __init__(self, documents: Iterable[Document],
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 *, revision: str | None = None, local_files_only: bool = False) -> None:
        self.documents = _documents(documents)
        _text(model_name, "model_name")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install ai-reliability-lab[multilingual] to use embeddings") from exc
        self._model = SentenceTransformer(model_name, revision=revision,
                                          local_files_only=local_files_only)
        self._embeddings = self._model.encode([d.text for d in self.documents],
                                              normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query: str, k: int = 5) -> tuple[str, ...]:
        _k(k)
        if not tokens(query):
            raise ValueError("query must contain searchable characters")
        vector = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self._embeddings @ vector
        order = sorted(range(len(self.documents)), key=lambda i: (-float(scores[i]), self.documents[i].id))
        return tuple(self.documents[i].id for i in order[:k])


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]
    language: str = "und"

    def __post_init__(self) -> None:
        if not tokens(self.query):
            raise ValueError("query must contain searchable characters")
        _text(self.language, "language")
        if isinstance(self.relevant_ids, str):
            raise ValueError("relevant_ids must be a collection of ids")
        object.__setattr__(self, "relevant_ids", frozenset(self.relevant_ids))
        if not self.relevant_ids:
            raise ValueError("retrieval cases need at least one relevant id")
        for doc_id in self.relevant_ids:
            _text(doc_id, "relevant id")


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


class RetrievalEvaluator:
    def __init__(self, retriever: Retriever, documents: Iterable[Document]) -> None:
        self.retriever = retriever
        self.document_ids = frozenset(d.id for d in _documents(documents))

    def evaluate(self, case: RetrievalCase, k: int = 5) -> RetrievalMetrics:
        _k(k)
        if not case.relevant_ids <= self.document_ids:
            raise ValueError("relevance labels contain unknown document ids")
        returned = self.retriever.search(case.query, k)
        if isinstance(returned, (str, bytes)) or not isinstance(returned, Sequence):
            raise ValueError("retriever must return a sequence of ids")
        ids = tuple(returned)
        if len(ids) > k or any(not isinstance(i, str) for i in ids):
            raise ValueError("invalid retriever output")
        if len(set(ids)) != len(ids) or not set(ids) <= self.document_ids:
            raise ValueError("retriever returned duplicate or unknown ids")
        hits = [int(doc_id in case.relevant_ids) for doc_id in ids]
        reciprocal = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(case.relevant_ids)) + 1))
        return RetrievalMetrics(sum(hits) / k, sum(hits) / len(case.relevant_ids), reciprocal, dcg / ideal)

    def evaluate_many(self, cases: Iterable[RetrievalCase], k: int = 5) -> RetrievalMetrics:
        _k(k)
        results = [self.evaluate(case, k) for case in cases]
        if not results:
            raise ValueError("at least one evaluation case is required")
        return RetrievalMetrics(*(sum(getattr(r, field) for r in results) / len(results)
                                  for field in RetrievalMetrics.__dataclass_fields__))


@dataclass(frozen=True)
class Claim:
    text: str
    citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.text, "claim")
        if isinstance(self.citation_ids, str):
            raise ValueError("citation_ids must be a collection")
        object.__setattr__(self, "citation_ids", tuple(self.citation_ids))
        for doc_id in self.citation_ids:
            _text(doc_id, "citation id")


@dataclass(frozen=True)
class GroundingReport:
    supported_fraction: float
    unsupported_claims: tuple[int, ...]
    unknown_citations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.unsupported_claims and not self.unknown_citations


def check_grounding(claims: Iterable[Claim], evidence: Iterable[Document]) -> GroundingReport:
    """Strict extractive support: contiguous normalized tokens in cited evidence.

    Evidence must be the retrieved context, not the entire corpus. This checks
    citation consistency, not factual truth, entailment, or safety.
    """
    docs = {doc.id: tokens(doc.text) for doc in _documents(evidence)}
    items = tuple(claims)
    if not items:
        raise ValueError("at least one claim is required; evaluate abstention separately")
    unsupported = []
    unknown = set()
    for index, claim in enumerate(items):
        needle = tokens(claim.text)
        unknown.update(doc_id for doc_id in claim.citation_ids if doc_id not in docs)
        supported = bool(needle) and any(
            any(haystack[start:start + len(needle)] == needle
                for start in range(len(haystack) - len(needle) + 1))
            for doc_id in claim.citation_ids if (haystack := docs.get(doc_id)) is not None)
        if not supported:
            unsupported.append(index)
    return GroundingReport((len(items) - len(unsupported)) / len(items), tuple(unsupported), tuple(sorted(unknown)))


@dataclass(frozen=True)
class AdversarialCase:
    """Human-labeled expected behavior for an injected, unsafe, or unanswerable query."""
    query: str
    must_abstain: bool
    forbidden_retrieval_ids: frozenset[str] = frozenset()
    category: str = "prompt_injection"

    def __post_init__(self) -> None:
        _text(self.query, "query")
        _text(self.category, "category")
        if not isinstance(self.must_abstain, bool):
            raise ValueError("must_abstain must be boolean")
        if isinstance(self.forbidden_retrieval_ids, str):
            raise ValueError("forbidden_retrieval_ids must be a collection")
        object.__setattr__(self, "forbidden_retrieval_ids", frozenset(self.forbidden_retrieval_ids))
        for doc_id in self.forbidden_retrieval_ids:
            _text(doc_id, "forbidden id")


@dataclass(frozen=True)
class AdversarialReport:
    behavior_matches: bool
    forbidden_hits: tuple[str, ...]
    grounding_passed: bool

    @property
    def passed(self) -> bool:
        return self.behavior_matches and not self.forbidden_hits and self.grounding_passed


def evaluate_adversarial(case: AdversarialCase, *, abstained: bool,
                         retrieved: Iterable[Document], claims: Iterable[Claim] = ()) -> AdversarialReport:
    """Score observed behavior. Abstention must come from structured application output."""
    if not isinstance(abstained, bool):
        raise ValueError("abstained must be boolean")
    docs, items = tuple(retrieved), tuple(claims)
    if docs:
        _documents(docs)
    if abstained and items:
        raise ValueError("an abstained response cannot contain claims")
    forbidden = tuple(sorted({d.id for d in docs} & case.forbidden_retrieval_ids))
    grounded = abstained or (bool(items) and bool(docs) and check_grounding(items, docs).passed)
    return AdversarialReport(abstained == case.must_abstain, forbidden, bool(grounded))
