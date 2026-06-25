"""
ngram_model.py
--------------
Modelos de lenguaje estadísticos basados en n-gramas para generar
versículos falsos, sin redes neuronales ni modelos preentrenados.

Se implementa una sola clase parametrizada por `n`, de forma que
unigram, bigram, trigram y "n-gram custom" son solo instancias distintas
de NGramModel (n=1, 2, 3, n).
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict

START = "<START>"
END = "<END>"


class NGramModel:
    def __init__(self, n: int):
        """
        n=1 -> unigram, n=2 -> bigram, n=3 -> trigram, n>=4 -> n-gram general.
        El contexto usado para predecir la siguiente palabra tiene tamaño n-1.
        """
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self.contexto_size = n - 1
        # conteos[contexto] -> Counter({palabra_siguiente: frecuencia})
        self.conteos: Dict[tuple, Counter] = defaultdict(Counter)
        self.vocabulario = set()

    def fit(self, oraciones_tokenizadas: List[List[str]]) -> "NGramModel":
        """oraciones_tokenizadas: lista de versículos, cada uno como lista de tokens (sin START/END)."""
        for tokens in oraciones_tokenizadas:
            secuencia = [START] * self.contexto_size + tokens + [END]
            self.vocabulario.update(tokens)
            for i in range(len(secuencia) - self.contexto_size):
                contexto = tuple(secuencia[i:i + self.contexto_size])
                siguiente = secuencia[i + self.contexto_size]
                self.conteos[contexto][siguiente] += 1
        return self

    def probabilidad(self, contexto: tuple, palabra: str) -> float:
        contador = self.conteos.get(contexto)
        if not contador:
            return 0.0
        total = sum(contador.values())
        return contador[palabra] / total

    def _siguiente_palabra(self, contexto: tuple) -> str:
        contador = self.conteos.get(contexto)
        if not contador:
            # contexto nunca visto -> backoff simple: elegir palabra aleatoria del vocabulario
            return random.choice(list(self.vocabulario)) if self.vocabulario else END
        palabras = list(contador.keys())
        pesos = list(contador.values())
        return random.choices(palabras, weights=pesos, k=1)[0]

    def generar(self, palabra_inicial: str = None, max_len: int = 30) -> str:
        """
        Genera una secuencia de al menos 15 palabras (max_len por defecto 30),
        deteniéndose si aparece <END> o al llegar a max_len.
        """
        if palabra_inicial:
            contexto = tuple([START] * (self.contexto_size - 1) + [palabra_inicial]) if self.contexto_size > 0 else tuple()
            generadas = [palabra_inicial]
        else:
            contexto = tuple([START] * self.contexto_size)
            generadas = []

        for _ in range(max_len):
            siguiente = self._siguiente_palabra(contexto)
            if siguiente == END:
                break
            generadas.append(siguiente)
            if self.contexto_size > 0:
                contexto = tuple((list(contexto) + [siguiente])[-self.contexto_size:])

        return " ".join(generadas)


def comparar_modelos(oraciones_tokenizadas: List[List[str]], ns=(1, 2, 3, 4), palabra_inicial=None, max_len=20):
    """Entrena un NGramModel por cada n y genera un texto de ejemplo de cada uno,
    para la comparación cualitativa pedida en el enunciado."""
    resultados = {}
    for n in ns:
        modelo = NGramModel(n).fit(oraciones_tokenizadas)
        resultados[n] = modelo.generar(palabra_inicial=palabra_inicial, max_len=max_len)
    return resultados
