"""
search_engine.py
-----------------
Buscador semántico: dado un texto (query del usuario o un versículo del
dataset), devuelve los K versículos más similares según similitud coseno
sobre vectores TF-IDF.
"""

from typing import List
import pandas as pd
import numpy as np

from .tfidf import TFIDFVectorizer, cosine_similarity
from .preprocessing import TextPreprocessor


class SemanticSearchEngine:
    def __init__(self, preprocessor: TextPreprocessor, vectorizer: TFIDFVectorizer):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.matriz_tfidf: np.ndarray = None
        self.df_corpus: pd.DataFrame = None  # debe tener columnas: libro, capitulo, versiculo, texto_original

    def fit(self, df_corpus: pd.DataFrame, columna_tokens: str = "texto_procesado"):
        """
        df_corpus debe contener una columna con los tokens ya procesados
        (lista de palabras) para cada versículo, además de metadatos
        (libro, capitulo, versiculo, texto_original).
        """
        self.df_corpus = df_corpus.reset_index(drop=True)
        documentos = df_corpus[columna_tokens].tolist()
        self.matriz_tfidf = self.vectorizer.fit_transform(documentos)
        return self

    def buscar(self, query: str, k: int = 5) -> pd.DataFrame:
        """Busca los k versículos más similares a un texto libre."""
        tokens_query = self.preprocessor.process(query)
        vector_query = self.vectorizer.vectorizar_texto_nuevo(tokens_query)
        return self._rankear(vector_query, k)

    def buscar_por_indice(self, idx_versiculo: int, k: int = 5) -> pd.DataFrame:
        """Busca los k versículos más similares a un versículo ya existente en el corpus."""
        vector_query = self.matriz_tfidf[idx_versiculo]
        resultados = self._rankear(vector_query, k + 1)  # +1 porque se incluirá a sí mismo
        return resultados[resultados.index != idx_versiculo].head(k)

    def _rankear(self, vector_query: np.ndarray, k: int) -> pd.DataFrame:
        similitudes = np.array([
            cosine_similarity(vector_query, self.matriz_tfidf[i])
            for i in range(self.matriz_tfidf.shape[0])
        ])
        top_idx = np.argsort(similitudes)[::-1][:k]

        resultado = self.df_corpus.iloc[top_idx].copy()
        resultado["similitud"] = similitudes[top_idx]
        columnas = [c for c in ["libro", "capitulo", "versiculo", "texto_original", "similitud"]
                    if c in resultado.columns]
        return resultado[columnas].reset_index(drop=True)
