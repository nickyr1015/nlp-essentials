from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import log, sqrt
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

DocumentLike = Union[str, Tuple[int, str], List[Union[int, str]], dict]
SparseVector = Dict[int, float]


def _extract_label_and_text(doc: DocumentLike) -> Tuple[int | None, str]:
    if isinstance(doc, dict):
        label = doc.get("label")
        text = doc.get("text", "")
        return (int(label) if label is not None else None, str(text))

    if isinstance(doc, (tuple, list)) and len(doc) >= 2:
        label, text = doc[0], doc[1]
        return (int(label) if label is not None else None, str(text))

    if isinstance(doc, str):
        parts = doc.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            return int(parts[0].strip()), parts[1].strip()
        return None, doc.strip()

    raise ValueError(f"Unsupported document format: {type(doc)}")


def _prepare_training_data(
    training_documents: Sequence[DocumentLike],
) -> Tuple[List[int], List[str]]:
    labels = []
    texts = []

    for doc in training_documents:
        label, text = _extract_label_and_text(doc)
        if label is None:
            raise ValueError("All training documents must have labels.")
        labels.append(label)
        texts.append(text)

    return labels, texts


def _prepare_test_data(
    test_documents: Sequence[DocumentLike],
) -> Tuple[List[int | None], List[str]]:
    labels = []
    texts = []

    for doc in test_documents:
        label, text = _extract_label_and_text(doc)
        labels.append(label)
        texts.append(text)

    return labels, texts


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[!?]", text.lower())
    negators = {"not", "no", "never", "n't"}
    negated_tokens = []

    for i, token in enumerate(tokens):
        negated_tokens.append(token)
        if token in negators and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token not in {"!", "?"}:
                negated_tokens.append(f"{token}_{next_token}")

    return negated_tokens


def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    padded = f" {text.lower()} "
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i : i + n])
    return ngrams


def build_vocabulary(
    documents: Sequence[str],
    min_df: int = 2,
    max_df_ratio: float = 0.95,
    feature_fn=tokenize,
) -> Dict[str, int]:
    df_counter = Counter()
    num_docs = len(documents)

    for doc in documents:
        unique_tokens = set(feature_fn(doc))
        for token in unique_tokens:
            df_counter[token] += 1

    vocab = {}
    for token, df in df_counter.items():
        if df >= min_df and df <= max_df_ratio * num_docs:
            vocab[token] = len(vocab)

    return vocab


def document_frequencies(
    vocab: Dict[str, int],
    documents: Sequence[str],
    feature_fn=tokenize,
) -> np.ndarray:
    dfs = np.zeros(len(vocab), dtype=float)

    for doc in documents:
        seen = set()
        for token in feature_fn(doc):
            if token in vocab and token not in seen:
                dfs[vocab[token]] += 1
                seen.add(token)

    return dfs


def tf_idf(
    vocab: Dict[str, int],
    dfs: np.ndarray,
    num_docs: int,
    document: str,
    feature_fn=tokenize,
) -> SparseVector:
    tf_counts = Counter(token for token in feature_fn(document) if token in vocab)
    vector = {}

    for token, count in tf_counts.items():
        idx = vocab[token]
        tf = 1.0 + log(count)
        idf = log((1.0 + num_docs) / (1.0 + dfs[idx])) + 1.0
        vector[idx] = tf * idf

    norm = sqrt(sum(value * value for value in vector.values()))
    if norm > 0:
        for idx in vector:
            vector[idx] /= norm

    return vector


def cosine_similarity(v1: SparseVector, v2: SparseVector) -> float:
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(value * v2.get(idx, 0.0) for idx, value in v1.items())


def blended_knn_predict(
    train_labels: Sequence[int],
    word_train_vectors: Sequence[SparseVector],
    char_train_vectors: Sequence[SparseVector],
    word_test_vector: SparseVector,
    char_test_vector: SparseVector,
    k: int,
    word_weight: float = 0.9,
) -> Tuple[int, float]:
    sims = []

    for label, word_train_vector, char_train_vector in zip(
        train_labels, word_train_vectors, char_train_vectors
    ):
        word_sim = cosine_similarity(word_test_vector, word_train_vector)
        char_sim = cosine_similarity(char_test_vector, char_train_vector)
        sim = word_weight * word_sim + (1.0 - word_weight) * char_sim
        sims.append((label, sim))

    sims.sort(key=lambda item: item[1], reverse=True)
    top_k = sims[:k]

    class_scores = defaultdict(float)
    class_counts = Counter()

    for label, sim in top_k:
        class_scores[label] += sim
        class_counts[label] += 1

    best_label = sorted(
        class_scores.keys(),
        key=lambda lbl: (-class_scores[lbl], -class_counts[lbl], lbl),
    )[0]

    return best_label, class_scores[best_label]


def knn_predict(
    train_vectors: List[Tuple[int, SparseVector]],
    test_vector: SparseVector,
    k: int,
) -> Tuple[int, float]:
    sims = []

    for label, train_vector in train_vectors:
        sim = cosine_similarity(test_vector, train_vector)
        sims.append((label, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]

    class_scores = defaultdict(float)
    class_counts = Counter()

    for label, sim in top_k:
        class_scores[label] += sim
        class_counts[label] += 1

    best_label = sorted(
        class_scores.keys(),
        key=lambda lbl: (-class_scores[lbl], -class_counts[lbl], lbl),
    )[0]

    return best_label, class_scores[best_label]


def sentiment_analyzer(
    training_documents: Sequence[DocumentLike],
    test_documents: Sequence[DocumentLike],
) -> List[Tuple[int, float]]:
    K = 75

    train_labels, train_texts = _prepare_training_data(training_documents)
    _, test_texts = _prepare_test_data(test_documents)

    vocab = build_vocabulary(train_texts, min_df=2, max_df_ratio=0.95)
    dfs = document_frequencies(vocab, train_texts)
    num_docs = len(train_texts)

    train_vectors = []
    for label, text in zip(train_labels, train_texts):
        vector = tf_idf(vocab, dfs, num_docs, text)
        train_vectors.append((label, vector))

    predictions = []
    for text in test_texts:
        test_vector = tf_idf(vocab, dfs, num_docs, text)
        pred_label, score = knn_predict(train_vectors, test_vector, K)
        predictions.append((pred_label, score))

    return predictions


def sentiment_analyzer_extra(
    training_documents: Sequence[DocumentLike],
    test_documents: Sequence[DocumentLike],
) -> List[Tuple[int, float]]:
    K = 137

    train_labels, train_texts = _prepare_training_data(training_documents)
    _, test_texts = _prepare_test_data(test_documents)

    word_vocab = build_vocabulary(train_texts, min_df=1, max_df_ratio=0.98)
    word_dfs = document_frequencies(word_vocab, train_texts)
    char_vocab = build_vocabulary(
        train_texts,
        min_df=2,
        max_df_ratio=0.98,
        feature_fn=char_ngrams,
    )
    char_dfs = document_frequencies(char_vocab, train_texts, feature_fn=char_ngrams)
    num_docs = len(train_texts)

    word_train_vectors = []
    char_train_vectors = []
    for text in train_texts:
        word_train_vectors.append(tf_idf(word_vocab, word_dfs, num_docs, text))
        char_train_vectors.append(
            tf_idf(
                char_vocab,
                char_dfs,
                num_docs,
                text,
                feature_fn=char_ngrams,
            )
        )

    predictions = []
    for text in test_texts:
        word_test_vector = tf_idf(word_vocab, word_dfs, num_docs, text)
        char_test_vector = tf_idf(
            char_vocab,
            char_dfs,
            num_docs,
            text,
            feature_fn=char_ngrams,
        )
        pred_label, score = blended_knn_predict(
            train_labels,
            word_train_vectors,
            char_train_vectors,
            word_test_vector,
            char_test_vector,
            K,
            0.85,
        )
        predictions.append((pred_label, score))

    return predictions