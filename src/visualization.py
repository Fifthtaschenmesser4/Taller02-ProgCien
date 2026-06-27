"""
visualization.py
Funciones para graficar cosas del dataset de versiculos.
Cada funcion arma un grafico y devuelve la figura.
Uso numpy, pandas y matplotlib. Solo en el PCA uso sklearn,
que el taller permite expresamente para esa parte.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_longitud_versiculos(df, columna="texto_original"):
    # largo de cada versiculo contando cuantas palabras tiene
    longitudes = [len(str(texto).split()) for texto in df[columna]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(longitudes, bins=40)
    ax.set_title("Distribucion de longitud de versiculos (en palabras)")
    ax.set_xlabel("Cantidad de palabras")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_versiculos_por_libro(df):
    # cuento cuantos versiculos hay por libro y ordeno de mayor a menor
    conteo = df.groupby("libro").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(conteo.index, conteo.values)
    ax.set_title("Cantidad de versiculos por libro")
    ax.set_ylabel("N° de versiculos")
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


def plot_heatmap_similitud_libros(matriz_similitud, nombres_libros):
    # heatmap NxN con la similitud entre libros
    fig, ax = plt.subplots(figsize=(18, 16))
    # "Blues": los valores bajos salen blancos y los altos azul oscuro
    imagen = ax.imshow(matriz_similitud, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    # pongo los nombres de los libros en los dos ejes.
    # uso letra chica (fontsize) porque son 66 nombres y si no no se leen
    ax.set_xticks(range(len(nombres_libros)))
    ax.set_yticks(range(len(nombres_libros)))
    ax.set_xticklabels(nombres_libros, rotation=90, fontsize=6)
    ax.set_yticklabels(nombres_libros, fontsize=6)

    ax.set_title("Similitud de coseno entre libros (basado en TF-IDF)")
    fig.tight_layout()
    return fig


def plot_pca_versiculos(matriz_tfidf, etiquetas, titulo="Versiculos proyectados con PCA"):
    # el taller permite usar sklearn para el PCA
    from sklearn.decomposition import PCA

    # si la matriz viene sparse la paso a densa, porque el PCA de sklearn no acepta sparse
    if hasattr(matriz_tfidf, "toarray"):
        matriz_tfidf = matriz_tfidf.toarray()

    # reduzco la matriz TF-IDF (muchas dimensiones) a solo 2 para poder graficarla
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(matriz_tfidf)

    # cuanta varianza explica cada componente (para ponerlo en los ejes)
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # armo un dataframe para graficar mas comodo
    if hasattr(etiquetas, "values"):
        etiquetas = etiquetas.values
    df_plot = pd.DataFrame({
        "PC1": componentes[:, 0],
        "PC2": componentes[:, 1],
        "etiqueta": etiquetas,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    # hago un scatter por cada etiqueta distinta (libro, testamento, etc),
    # asi cada grupo queda de un color
    for etiqueta in df_plot["etiqueta"].unique():
        sub = df_plot[df_plot["etiqueta"] == etiqueta]
        ax.scatter(sub["PC1"], sub["PC2"], s=15, alpha=0.5, label=etiqueta)

    ax.set_title(titulo)
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% var. explicada)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% var. explicada)")

    # si hay muchas etiquetas la leyenda no se entiende, asi que la omito
    if df_plot["etiqueta"].nunique() <= 15:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_sentimiento_por_libro(df_sentimiento_agregado):
    # ordeno por sentimiento para que las barras se lean como un ranking
    df_ord = df_sentimiento_agregado.sort_values("mean")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_ord["libro"], df_ord["mean"])
    ax.set_title("Sentimiento promedio por libro")
    ax.set_xlabel("Sentimiento promedio (negativo <- 0 -> positivo)")
    fig.tight_layout()
    return fig


def plot_wordcloud(frecuencias, titulo="Palabras mas frecuentes"):
    # frecuencias es un diccionario {palabra: cantidad}
    from wordcloud import WordCloud

    wc = WordCloud(width=900, height=500, background_color="white")
    wc = wc.generate_from_frequencies(frecuencias)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo)
    return fig


def plot_matriz_confusion(cm, clases):
    fig, ax = plt.subplots(figsize=(14, 12))
    imagen = ax.imshow(cm, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=90)
    ax.set_yticklabels(clases)

    ax.set_title("Matriz de confusion - Clasificador de versiculos")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig