"""
sentiment.py
------------
Analisis de sentimiento por versiculo y agregacion por capitulo/libro.

Se define una interfaz SentimentAnalyzer con dos implementaciones:
- LexiconSentimentAnalyzer: basado en diccionario propio (pensado para
  corpus en espanol, sin depender de librerias externas).
- TextBlobSentimentAnalyzer: usa la libreria TextBlob (recomendada si
  el corpus esta en ingles).
"""

import pandas as pd


class SentimentAnalyzer:
    """
    Interfaz base para los analizadores de sentimiento.

    Cualquier analizador debe implementar el metodo score(texto), que
    devuelve un valor aproximado entre -1 (negativo) y 1 (positivo).
    """

    def score(self, texto):
        """
        Calcula el puntaje de sentimiento de un texto.
        Args:
            texto (str): Texto a analizar (ej. un versiculo).
        Returns:
            float: Puntaje de sentimiento, aproximadamente entre -1 y 1.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    """
    Clase que representa un analizador de sentimiento basado en diccionario.

    Pensado como alternativa simple para corpus en espanol, donde
    librerias como TextBlob/VADER (entrenadas en ingles) no aplican bien.
    Se recomienda ampliar las listas o reemplazar por un lexicon en
    espanol mas completo (ej. SEL, Spanish Emotion Lexicon) si se
    necesita mayor precision.

    Attributes:
        palabras_positivas (set[str]): Palabras consideradas positivas.
        palabras_negativas (set[str]): Palabras consideradas negativas.
    """

    PALABRAS_POSITIVAS_BASE = {
        "amor", "paz", "alegria", "alegria", "bendicion", "bendicion", "gracia",
        "fe", "esperanza", "gloria", "salvacion", "salvacion", "bueno", "buena",
        "bendito", "bendita", "gozo", "misericordia", "luz", "vida", "justicia",
        "victoria", "fiel", "fidelidad", "consuelo", "verdad", "santo", "santa",
    }
    PALABRAS_NEGATIVAS_BASE = {
        "muerte", "pecado", "maldicion", "maldicion", "ira", "guerra", "dolor",
        "mal", "maldad", "destruccion", "destruccion", "castigo", "miedo",
        "temor", "llanto", "lamento", "oscuridad", "enemigo", "violencia",
        "sangre", "venganza", "juicio", "condenacion", "condenacion", "infierno",
    }

    def __init__(self, palabras_positivas=None, palabras_negativas=None):
        """
        Inicializa el analizador con un diccionario de palabras positivas/negativas.
        Args:
            palabras_positivas (set[str], optional): Reemplaza la lista base
                de palabras positivas si se entrega.
            palabras_negativas (set[str], optional): Reemplaza la lista base
                de palabras negativas si se entrega.
        """
        self.palabras_positivas = palabras_positivas if palabras_positivas is not None else set(self.PALABRAS_POSITIVAS_BASE)
        self.palabras_negativas = palabras_negativas if palabras_negativas is not None else set(self.PALABRAS_NEGATIVAS_BASE)

    def agregar_palabra_positiva(self, palabra):
        """
        Agrega una palabra al diccionario de palabras positivas.
        Args:
            palabra (str): Palabra a agregar (en minusculas).
        """
        self.palabras_positivas.add(palabra)

    def agregar_palabra_negativa(self, palabra):
        """
        Agrega una palabra al diccionario de palabras negativas.
        Args:
            palabra (str): Palabra a agregar (en minusculas).
        """
        self.palabras_negativas.add(palabra)

    def score(self, texto):
        """
        Calcula el sentimiento de un texto como (positivas - negativas) / total.
        Args:
            texto (str): Texto a analizar.
        Returns:
            float: Puntaje entre -1 y 1. Devuelve 0.0 si el texto esta vacio
                o no contiene palabras del diccionario.
        """
        tokens = texto.lower().split()
        if not tokens:
            return 0.0
        positivas = sum(1 for t in tokens if t in self.palabras_positivas)
        negativas = sum(1 for t in tokens if t in self.palabras_negativas)
        total = positivas + negativas
        if total == 0:
            return 0.0
        return (positivas - negativas) / total


class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    """
    Clase que representa un analizador de sentimiento basado en TextBlob.
    Recomendada si el corpus elegido esta en ingles.

    Attributes:
        text_blob_class (type): Referencia a la clase TextBlob, importada
            en el constructor para no forzar la dependencia si no se usa.
    """

    def __init__(self):
        """Inicializa el analizador, importando TextBlob de forma local."""
        from textblob import TextBlob
        self.text_blob_class = TextBlob

    def score(self, texto):
        """
        Calcula la polaridad de un texto usando TextBlob.
        Args:
            texto (str): Texto a analizar.
        Returns:
            float: Polaridad entre -1 (negativo) y 1 (positivo).
        """
        return self.text_blob_class(texto).sentiment.polarity


def calcular_sentimiento_corpus(df, analyzer, columna_texto="texto_original"):
    """
    Agrega una columna 'sentimiento' al DataFrame de versiculos.
    Args:
        df (pd.DataFrame): DataFrame de versiculos.
        analyzer (SentimentAnalyzer): Analizador a usar (ej. LexiconSentimentAnalyzer).
        columna_texto (str): Columna que contiene el texto a analizar.
    Returns:
        pd.DataFrame: Copia de df con la columna 'sentimiento' agregada.
    """
    df = df.copy()
    df["sentimiento"] = df[columna_texto].apply(analyzer.score)
    return df


def agregar_por_libro(df):
    """
    Agrega los puntajes de sentimiento por libro.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'libro' y 'sentimiento'.
    Returns:
        pd.DataFrame: Una fila por libro, con mean, std, min, max y count
            del sentimiento, ordenado de menor a mayor sentimiento promedio.
    """
    return (
        df.groupby("libro")["sentimiento"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )


def agregar_por_capitulo(df):
    """
    Agrega los puntajes de sentimiento por libro y capitulo.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'libro', 'capitulo' y 'sentimiento'.
    Returns:
        pd.DataFrame: Una fila por (libro, capitulo) con el sentimiento promedio.
    """
    return (
        df.groupby(["libro", "capitulo"])["sentimiento"]
        .mean()
        .reset_index()
    )