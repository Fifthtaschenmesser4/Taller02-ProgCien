import math
from collections import Counter
from typing import List, Dict
import numpy as np


class TFIDFVectorizer:
    """
    TF  (term frequency):       tf(t, d) = count(t in d) / len(d)
    IDF (inverse doc frequency): idf(t)   = log( (1 + N) / (1 + df(t)) ) + 1
        (suavizado +1 estilo sklearn para evitar división por cero)
    TF-IDF(t, d) = tf(t, d) * idf(t)
    """

    def __init__(self):
        self.vocabulario: Dict[str, int] = {}
        self.idf: np.ndarray = None
        self.n_docs: int = 0

    def fit(self, documentos_tokenizados: List[List[str]]) -> "TFIDFVectorizer":
        self.n_docs = len(documentos_tokenizados)

        vocab_set = set()
        for tokens in documentos_tokenizados:
            vocab_set.update(tokens)

        self.vocabulario = {palabra: i for i, palabra in enumerate(sorted(vocab_set))}

        df = np.zeros(len(self.vocabulario))
        for tokens in documentos_tokenizados:
            presentes = set(tokens)
            for palabra in presentes:
                df[self.vocabulario[palabra]] += 1

        self.idf = np.log((1 + self.n_docs) / (1 + df)) + 1
        return self

    def transform(self, documentos_tokenizados: List[List[str]]) -> np.ndarray:
        """Devuelve matriz (n_documentos x n_vocabulario) con pesos TF-IDF."""
        
        if self.idf is None:
            raise RuntimeError("Llama a fit() antes de transform().")

        matriz = np.zeros((len(documentos_tokenizados), len(self.vocabulario)))
        for i, tokens in enumerate(documentos_tokenizados):
            if not tokens:
                continue
            conteo = Counter(tokens)
            total = len(tokens)
            for palabra, freq in conteo.items():
                if palabra in self.vocabulario:
                    j = self.vocabulario[palabra]
                    tf = freq / total
                    matriz[i, j] = tf * self.idf[j]
        return matriz

    def fit_transform(self, documentos_tokenizados: List[List[str]]) -> np.ndarray:
        self.fit(documentos_tokenizados)
        return self.transform(documentos_tokenizados)

    def vectorizar_texto_nuevo(self, tokens: List[str]) -> np.ndarray:
        """Vectoriza un único texto nuevo (ej: query del buscador semántico)
        usando el vocabulario/idf ya ajustados, sin reentrenar."""
        return self.transform([tokens])[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    similitud(A, B) = (A . B) / (||A|| * ||B||)
    Implementación propia, sin sklearn.metrics.pairwise.
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = math.sqrt(np.dot(vec_a, vec_a))
    norm_b = math.sqrt(np.dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_matrix(matriz: np.ndarray) -> np.ndarray:
    """
    Similitud de coseno entre todas las filas de una matriz (NxN).
    Útil para el heatmap de similitud entre libros pedido en 3.2.
    Vectorizado para que sea rápido sobre corpus grandes.
    """
    normas = np.linalg.norm(matriz, axis=1)
    normas[normas == 0] = 1e-10  # evita división por cero
    matriz_normalizada = matriz / normas[:, np.newaxis]
    return matriz_normalizada @ matriz_normalizada.T
