"""
preprocessing.py
----------------
Pipeline de preprocesamiento textual encapsulado en una clase para que
sea fácil justificar/activar-desactivar etapas (pedido explícito en el
enunciado: "justificar las decisiones tomadas").
"""

import re
import string
from collections import Counter
from typing import List, Dict, Iterable, Optional


# Lista mínima de stopwords en español. Se recomienda complementar con
# nltk.corpus.stopwords('spanish') o spacy si lo tienen disponible;
# se deja una lista base para no depender de descargas externas.
STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas",
    "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías",
    "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "es", "fue", "ha", "han", "será", "son", "era", "eran", "sea", "sean",
}


class TextPreprocessor:
    """
    Encapsula cada etapa del pipeline como un método independiente para
    poder activarlas/desactivarlas y medir su impacto individualmente
    (pedido en la sección 3.8 del enunciado: "Impacto del preprocesamiento
    sobre los resultados finales").
    """

    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
    ):
        self.stopwords = set(stopwords) if stopwords is not None else STOPWORDS_ES
        self.min_token_len = min_token_len
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.vocabulario: Dict[str, int] = {}
        self.frecuencias: Counter = Counter()

    # ---- etapas individuales ----

    def to_lowercase(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def strip_punctuation(self, text: str) -> str:
        if not self.remove_punctuation:
            return text
        # incluye signos propios de español (¿¡) además de string.punctuation
        punct = string.punctuation + "¿¡“”‘’—–"
        return text.translate(str.maketrans(punct, " " * len(punct)))

    def strip_special_and_numbers(self, text: str) -> str:
        if not self.remove_numbers:
            return text
        # conserva letras (incl. acentos/ñ) y espacios
        return re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", " ", text)

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def filter_short_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if len(t) >= self.min_token_len]

    # ---- pipeline completo ----

    def process(self, text: str) -> List[str]:
        """Aplica todas las etapas en orden y devuelve la lista de tokens."""
        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.strip_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.filter_stopwords(tokens)
        tokens = self.filter_short_tokens(tokens)
        return tokens

    def process_corpus(self, textos: List[str]) -> List[List[str]]:
        """Procesa una lista de textos y actualiza vocabulario/frecuencias globales."""
        resultado = []
        for texto in textos:
            tokens = self.process(texto)
            resultado.append(tokens)
            self.frecuencias.update(tokens)
        self.vocabulario = {palabra: idx for idx, palabra in enumerate(sorted(self.frecuencias))}
        return resultado

    def palabras_mas_frecuentes(self, n: int = 20):
        return self.frecuencias.most_common(n)
