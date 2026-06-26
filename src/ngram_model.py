"""
ngram_model.py
--------------
Modelos de lenguaje estadisticos basados en n-gramas para generar
versiculos falsos, sin redes neuronales ni modelos preentrenados.

Se implementa una sola clase parametrizada por "n", de forma que
unigram, bigram, trigram y "n-gram custom" son solo instancias distintas
de NGramModel (n=1, 2, 3, n).
"""

import random
from collections import defaultdict, Counter

START = "<START>"
END = "<END>"


class NGramModel:
    """
    Clase que representa un modelo de lenguaje basado en n-gramas.

    Attributes:
        n (int): Tamano del n-grama (1=unigram, 2=bigram, 3=trigram, etc).
        contexto_size (int): Cantidad de palabras de contexto usadas para
            predecir la siguiente palabra (n - 1).
        conteos (dict[tuple, Counter]): Para cada contexto (tupla de
            palabras), un contador de cuantas veces aparecio cada palabra
            siguiente en el corpus de entrenamiento.
        vocabulario (set[str]): Conjunto de palabras vistas durante el
            entrenamiento (sin incluir <START>/<END>).
    """

    def __init__(self, n):
        """
        Inicializa el modelo de n-gramas.
        Args:
            n (int): Tamano del n-grama. Debe ser mayor o igual a 1.
        """
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self.contexto_size = n - 1
        self.conteos = defaultdict(Counter)
        self.vocabulario = set()

    def fit(self, oraciones_tokenizadas):
        """
        Entrena el modelo contando ocurrencias de cada n-grama en el corpus.
        Args:
            oraciones_tokenizadas (list[list[str]]): Lista de versiculos,
                cada uno representado como lista de tokens (sin <START>/<END>).
        Returns:
            NGramModel: La misma instancia, para poder encadenar llamadas.
        """
        for tokens in oraciones_tokenizadas:
            secuencia = [START] * self.contexto_size + tokens + [END]
            self.vocabulario.update(tokens)
            for i in range(len(secuencia) - self.contexto_size):
                contexto = tuple(secuencia[i:i + self.contexto_size])
                siguiente = secuencia[i + self.contexto_size]
                self.conteos[contexto][siguiente] += 1
        return self

    def get_probabilidad(self, contexto, palabra):
        """
        Calcula la probabilidad empirica de que "palabra" siga a "contexto".
        Args:
            contexto (tuple[str]): Tupla de palabras de contexto (tamano contexto_size).
            palabra (str): Palabra candidata a continuar la secuencia.
        Returns:
            float: Probabilidad estimada (frecuencia relativa), entre 0 y 1.
        """
        contador = self.conteos.get(contexto)
        if not contador:
            return 0.0
        total = sum(contador.values())
        return contador[palabra] / total

    def get_siguiente_palabra(self, contexto):
        """
        Elige la siguiente palabra dado un contexto, muestreando segun
        las frecuencias observadas durante el entrenamiento.
        Args:
            contexto (tuple[str]): Tupla de palabras de contexto.
        Returns:
            str: Palabra elegida (puede ser <END>), o una palabra aleatoria
                del vocabulario si el contexto nunca fue visto (backoff simple).
        """
        contador = self.conteos.get(contexto)
        if not contador:
            return random.choice(list(self.vocabulario)) if self.vocabulario else END
        palabras = list(contador.keys())
        pesos = list(contador.values())
        return random.choices(palabras, weights=pesos, k=1)[0]

    def generar(self, palabra_inicial=None, max_len=30):
        """
        Genera una secuencia de palabras a partir del modelo entrenado.
        Args:
            palabra_inicial (str, optional): Palabra desde la que comenzar
                la generacion. Si es None, se comienza desde el contexto inicial <START>.
            max_len (int): Cantidad maxima de palabras a generar (la generacion
                tambien se detiene antes si aparece el token <END>).
        Returns:
            str: Texto generado, con las palabras separadas por espacios.
        """
        if palabra_inicial:
            if self.contexto_size > 0:
                contexto = tuple([START] * (self.contexto_size - 1) + [palabra_inicial])
            else:
                contexto = tuple()
            generadas = [palabra_inicial]
        else:
            contexto = tuple([START] * self.contexto_size)
            generadas = []

        for _ in range(max_len):
            siguiente = self.get_siguiente_palabra(contexto)
            if siguiente == END:
                break
            generadas.append(siguiente)
            if self.contexto_size > 0:
                contexto = tuple((list(contexto) + [siguiente])[-self.contexto_size:])

        return " ".join(generadas)

    def __str__(self):
        return f"NGramModel(n={self.n}, vocabulario={len(self.vocabulario)} palabras)"


def comparar_modelos(oraciones_tokenizadas, ns=(1, 2, 3, 4), palabra_inicial=None, max_len=20):
    """
    Entrena un NGramModel por cada valor de n y genera un texto de ejemplo
    de cada uno, para la comparacion cualitativa pedida en el enunciado.
    Args:
        oraciones_tokenizadas (list[list[str]]): Corpus tokenizado (versiculos).
        ns (tuple[int]): Valores de n a comparar (ej. unigram, bigram, trigram, n custom).
        palabra_inicial (str, optional): Palabra inicial para todas las generaciones.
        max_len (int): Largo maximo de cada texto generado.
    Returns:
        dict[int, str]: Para cada n, el texto generado por su NGramModel.
    """
    resultados = {}
    for n in ns:
        modelo = NGramModel(n).fit(oraciones_tokenizadas)
        resultados[n] = modelo.generar(palabra_inicial=palabra_inicial, max_len=max_len)
    return resultados