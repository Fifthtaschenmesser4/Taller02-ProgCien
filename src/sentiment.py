"""
sentiment.py
------------
Análisis de sentimiento por versículo y agregación por capítulo/libro.

Usa un enfoque basado en léxico (ej: librería `pysentimiento` o un
diccionario propio) ya que herramientas como TextBlob/VADER están
entrenadas en inglés y el corpus probablemente esté en español.

Si trabajan en inglés con el dataset de Kaggle, pueden cambiar
`AnalizadorLexico` por TextBlob directamente (más simple).
Se deja una interfaz `SentimentAnalyzer` para que sea intercambiable.
"""

from typing import List, Dict
import pandas as pd


class SentimentAnalyzer:
    """
    Interfaz mínima: cualquier analizador debe implementar `score(texto) -> float`
    en rango aproximado [-1, 1] (negativo a positivo).
    """

    def score(self, texto: str) -> float:
        raise NotImplementedError


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    """
    Analizador simple basado en diccionario de palabras positivas/negativas.
    Pensado como fallback si no quieren depender de librerías externas o si
    el corpus está en español y las librerías en inglés no aplican bien.
    Se recomienda ampliar las listas o reemplazar por un lexicón en español
    como el de SEL (Spanish Emotion Lexicon) si quieren más precisión.
    """

    POSITIVAS = {
        "amor", "paz", "alegria", "alegría", "bendicion", "bendición", "gracia",
        "fe", "esperanza", "gloria", "salvacion", "salvación", "bueno", "buena",
        "bendito", "bendita", "gozo", "misericordia", "luz", "vida", "justicia",
        "victoria", "fiel", "fidelidad", "consuelo", "verdad", "santo", "santa",
    }
    NEGATIVAS = {
        "muerte", "pecado", "maldicion", "maldición", "ira", "guerra", "dolor",
        "mal", "maldad", "destruccion", "destrucción", "castigo", "miedo",
        "temor", "llanto", "lamento", "oscuridad", "enemigo", "violencia",
        "sangre", "venganza", "juicio", "condenacion", "condenación", "infierno",
    }

    def score(self, texto: str) -> float:
        tokens = texto.lower().split()
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in self.POSITIVAS)
        neg = sum(1 for t in tokens if t in self.NEGATIVAS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total


class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    """Alternativa usando TextBlob (recomendada si el corpus está en inglés)."""

    def __init__(self):
        from textblob import TextBlob  # import local para no forzar la dependencia si no se usa
        self._TextBlob = TextBlob

    def score(self, texto: str) -> float:
        return self._TextBlob(texto).sentiment.polarity


def calcular_sentimiento_corpus(
    df: pd.DataFrame,
    analyzer: SentimentAnalyzer,
    columna_texto: str = "texto_original",
) -> pd.DataFrame:
    """Agrega una columna 'sentimiento' al DataFrame de versículos."""
    df = df.copy()
    df["sentimiento"] = df[columna_texto].apply(analyzer.score)
    return df


def agregar_por_libro(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("libro")["sentimiento"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )


def agregar_por_capitulo(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["libro", "capitulo"])["sentimiento"]
        .mean()
        .reset_index()
    )
