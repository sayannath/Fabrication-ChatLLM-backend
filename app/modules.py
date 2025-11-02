import csv
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dspy
import structlog
import weave  # type: ignore

logger = structlog.get_logger(__name__)

WEAVE_USERNAME = "sayannath235"
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "fabrication-rag")
WEAVE_PATH = f"{WEAVE_USERNAME}/{WEAVE_PROJECT}"

weave.init(project_name=WEAVE_PATH)


@weave.op()
def _log_retrieval(question: str, contexts: Sequence[str], scores: Sequence[float]):
    return {
        "question": question,
        "contexts": list(contexts),
        "scores": list(scores),
    }


@weave.op()
def _log_generation(question: str, answer: str, context_block: str):
    return {"question": question, "answer": answer, "context_block": context_block}


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "LLM-Dataset - Fabrication.csv"


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass
class FabricationDocument:
    title: str
    text: str
    metadata: Dict[str, str]

    def snippet(self, limit: int = 240) -> str:
        return self.text[:limit]


class FabricationDataset:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.documents: List[FabricationDocument] = self._load_documents()

    def _load_documents(self) -> List[FabricationDocument]:
        if not self.csv_path.exists():
            msg = f"Dataset not found at {self.csv_path}"
            logger.error("dataset.missing", path=str(self.csv_path))
            raise FileNotFoundError(msg)

        documents: List[FabricationDocument] = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                if not any(value.strip() for value in row.values() if value):
                    continue
                title = row.get("Paper name") or f"Row {idx + 1}"
                text_parts = [
                    f"{key}: {value.strip()}"
                    for key, value in row.items()
                    if value and value.strip()
                ]
                text = "\n".join(text_parts)
                documents.append(
                    FabricationDocument(title=title, text=text, metadata=row)
                )
        logger.info(
            "dataset.loaded",
            path=str(self.csv_path),
            documents=len(documents),
        )
        return documents


class SimpleBM25Retriever:
    def __init__(
        self,
        documents: Sequence[FabricationDocument],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokenized_docs: List[List[str]] = [_tokenize(doc.text) for doc in self.documents]
        self._doc_freq: Dict[str, int] = {}
        for tokens in self._tokenized_docs:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avg_doc_len = (
            sum(len(tokens) for tokens in self._tokenized_docs) / len(self._tokenized_docs)
            if self._tokenized_docs
            else 0.0
        )

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0) + 0.5
        return math.log((len(self.documents) - df + 0.5) / df + 1.0)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[FabricationDocument, float]]:
        if not self.documents:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, tokens in enumerate(self._tokenized_docs):
            doc_len = len(tokens) or 1
            token_freq: Dict[str, int] = {}
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            avg_doc_len = self._avg_doc_len or 1.0

            score = 0.0
            for term in query_tokens:
                if term not in token_freq:
                    continue
                tf = token_freq[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
                score += self._idf(term) * numerator / (denominator or 1)
            scores.append((idx, score))

        scored_docs = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.documents[idx], score) for idx, score in scored_docs if score > 0]


class RAGSignature(dspy.Signature):
    contexts = dspy.InputField(
        desc="Relevant fabrication research snippets that can help answer the question."
    )
    question = dspy.InputField(desc="The user's question about fabrication research devices.")
    answer = dspy.OutputField(desc="A grounded answer citing the retrieved evidence.")


@lru_cache(maxsize=1)
def _get_dataset(csv_path: Path) -> FabricationDataset:
    return FabricationDataset(csv_path=csv_path)


class ChainOfThought(dspy.Module):
    """Retrieval-augmented reasoning over the fabrication dataset."""

    def __init__(self, dataset_path: Path | None = None, top_k: int = 3):
        super().__init__()
        env_override = os.getenv("FABRICATION_DATASET_PATH")
        if env_override:
            resolved_path = Path(env_override).expanduser()
        elif dataset_path is not None:
            resolved_path = Path(dataset_path).expanduser()
        else:
            resolved_path = DEFAULT_DATASET
        self.dataset = _get_dataset(resolved_path)
        self.retriever = SimpleBM25Retriever(self.dataset.documents)
        self.top_k = top_k
        self.generator = dspy.Predict(RAGSignature)

    def forward(self, question: str):
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [doc.text for doc, _ in retrieved]
        context_block = "\n\n---\n\n".join(contexts) if contexts else ""
        scores = [score for _, score in retrieved]

        _log_retrieval(question, contexts, scores)

        generator_context = (
            context_block
            if context_block
            else "No relevant entries found in the fabrication dataset; respond using general knowledge."
        )

        prediction = self.generator(question=question, contexts=generator_context)
        answer = getattr(prediction, "answer", "")

        _log_generation(question, answer, generator_context)

        prediction.contexts = contexts
        prediction.sources = [
            {
                "paper": doc.title,
                "score": score,
                "snippet": doc.snippet(),
                "metadata": doc.metadata,
            }
            for doc, score in retrieved
        ]
        return prediction
