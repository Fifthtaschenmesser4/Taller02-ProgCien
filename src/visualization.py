"""
visualization.py
-----------------
Funciones de visualización reutilizables. Se mantienen como funciones
(no clases) porque cada una es independiente y se usan puntualmente desde
el notebook/main, pero se agrupan en un módulo para mantener el proyecto
organizado.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_longitud_versiculos(df: pd.DataFrame, columna: str = "texto_original"):
    longitudes = df[columna].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(longitudes, bins=40, ax=ax)
    ax.set_title("Distribución de longitud de versículos (en palabras)")
    ax.set_xlabel("Cantidad de palabras")
    ax.set_ylabel("Frecuencia")
    return fig


def plot_versiculos_por_libro(df: pd.DataFrame):
    conteo = df.groupby("libro").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    conteo.plot(kind="bar", ax=ax)
    ax.set_title("Cantidad de versículos por libro")
    ax.set_ylabel("N° de versículos")
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


def plot_heatmap_similitud_libros(matriz_similitud: np.ndarray, nombres_libros: List[str]):
    """Heatmap NxN obligatorio según el enunciado (sección 3.2)."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matriz_similitud,
        xticklabels=nombres_libros,
        yticklabels=nombres_libros,
        cmap="viridis",
        ax=ax,
    )
    ax.set_title("Similitud de coseno entre libros (basado en TF-IDF)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_pca_versiculos(
    matriz_tfidf: np.ndarray,
    etiquetas: pd.Series,
    titulo: str = "Versículos proyectados con PCA",
):
    """
    Reduce la matriz TF-IDF a 2D con PCA (sklearn permitido aquí) y
    grafica cada versículo coloreado según `etiquetas` (libro, testamento, etc).
    """
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(matriz_tfidf)

    df_plot = pd.DataFrame({
        "PC1": componentes[:, 0],
        "PC2": componentes[:, 1],
        "etiqueta": etiquetas.values if hasattr(etiquetas, "values") else etiquetas,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="etiqueta", alpha=0.5, s=15, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var. explicada)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var. explicada)")
    if df_plot["etiqueta"].nunique() > 15:
        ax.legend([], [], frameon=False)  # demasiadas categorías -> se omite leyenda
    fig.tight_layout()
    return fig, pca


def plot_sentimiento_por_libro(df_sentimiento_agregado: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_sentimiento_agregado["libro"], df_sentimiento_agregado["mean"])
    ax.set_title("Sentimiento promedio por libro")
    ax.set_xlabel("Sentimiento promedio (negativo <- 0 -> positivo)")
    fig.tight_layout()
    return fig


def plot_wordcloud(frecuencias: dict, titulo: str = "Palabras más frecuentes"):
    from wordcloud import WordCloud
    wc = WordCloud(width=900, height=500, background_color="white").generate_from_frequencies(frecuencias)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo)
    return fig


def plot_matriz_confusion(cm: np.ndarray, clases: List[str]):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=clases, yticklabels=clases, annot=False, cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusión - Clasificador de versículos")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig
