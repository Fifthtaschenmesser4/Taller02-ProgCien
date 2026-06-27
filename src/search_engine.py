"""
search_engine.py
Buscador semantico: dado un texto (lo que escribe el usuario o un versiculo
del dataset), devuelve los K versiculos mas parecidos usando similitud coseno
sobre los vectores TF-IDF.
"""

import pandas as pd
import numpy as np

from .tfidf import TFIDFVectorizer, cosine_similarity
from .preprocessing import TextPreprocessor


class SemanticSearchEngine:

    def __init__(self, preprocessor, vectorizer):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.matriz_tfidf = None
        self.df_corpus = None   # debe tener columnas: libro, capitulo, versiculo, texto_original

    def fit(self, df_corpus, columna_tokens="texto_procesado"):
        # guardo el corpus y calculo la matriz TF-IDF de todos los versiculos.
        # columna_tokens es la columna con los tokens ya procesados.
        self.df_corpus = df_corpus.reset_index(drop=True)
        documentos = df_corpus[columna_tokens].tolist()
        self.matriz_tfidf = self.vectorizer.fit_transform(documentos)
        return self

    def buscar(self, query, k=5):
        # busco los k versiculos mas parecidos a un texto que escribe el usuario.
        # proceso el texto -> lo paso a vector TF-IDF -> rankeo.
        tokens_query = self.preprocessor.process(query)
        vector_query = self.vectorizer.vectorizar_texto_nuevo(tokens_query)
        resultado = self._rankear(vector_query, k)
        return resultado.reset_index(drop=True)

    def buscar_por_indice(self, idx_versiculo, k=5):
        # busco los k mas parecidos a un versiculo que YA esta en el corpus.
        # pido k+1 porque el mismo versiculo sale primero (similitud 1) y despues lo saco.
        vector_query = self.matriz_tfidf[idx_versiculo]
        resultado = self._rankear(vector_query, k + 1)
        # saco el propio versiculo usando su indice real del corpus
        resultado = resultado[resultado.index != idx_versiculo].head(k)
        return resultado.reset_index(drop=True)

    def _rankear(self, vector_query, k):
        # calculo la similitud coseno del query contra cada versiculo del corpus
        similitudes = np.array([
            cosine_similarity(vector_query, self.matriz_tfidf[i])
            for i in range(self.matriz_tfidf.shape[0])
        ])

        # argsort ordena de menor a mayor y devuelve los indices;
        # lo doy vuelta con [::-1] para tener de mayor a menor y me quedo con los k primeros
        top_idx = np.argsort(similitudes)[::-1][:k]

        # armo la tabla con los metadatos + la similitud.
        # OJO: aca NO reseteo el indice, asi mantengo el indice original del corpus
        # (lo necesita buscar_por_indice para sacar el propio versiculo).
        resultado = self.df_corpus.iloc[top_idx].copy()
        resultado["similitud"] = similitudes[top_idx]

        columnas = [c for c in ["libro", "capitulo", "versiculo", "texto_original", "similitud"]
                    if c in resultado.columns]
        return resultado[columnas]