"""
classifier.py
--------------
Clasificador que predice el libro al que pertenece un versículo, usando
como features la matriz TF-IDF (implementada a mano en tfidf.py) y como
modelo un clasificador estándar de sklearn (esto SÍ está permitido, la
restricción del enunciado es solo sobre TF-IDF y similitud coseno).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class VerseClassifier:
    def __init__(self, modelo: str = "logistic"):
        """
        modelo: "logistic" | "naive_bayes"
        Se deja Naive Bayes como alternativa simple/rápida para comparar,
        ya que es habitual como baseline en clasificación de texto.
        """
        if modelo == "logistic":
            self.modelo = LogisticRegression(max_iter=1000)
        elif modelo == "naive_bayes":
            self.modelo = MultinomialNB()
        else:
            raise ValueError("modelo debe ser 'logistic' o 'naive_bayes'")

        self.nombre_modelo = modelo
        self.clases_ = None

    def entrenar(self, X: np.ndarray, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        # MultinomialNB requiere features no-negativas; TF-IDF ya lo cumple.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.modelo.fit(X_train, y_train)
        self.clases_ = self.modelo.classes_
        self._X_test, self._y_test = X_test, y_test
        return X_train, X_test, y_train, y_test

    def evaluar(self) -> dict:
        y_pred = self.modelo.predict(self._X_test)
        acc = accuracy_score(self._y_test, y_pred)
        cm = confusion_matrix(self._y_test, y_pred, labels=self.clases_)
        reporte = classification_report(self._y_test, y_pred, labels=self.clases_, zero_division=0)
        return {
            "accuracy": acc,
            "matriz_confusion": cm,
            "clases": self.clases_,
            "reporte": reporte,
            "y_test": self._y_test,
            "y_pred": y_pred,
        }

    def predecir(self, X_nuevo: np.ndarray) -> np.ndarray:
        return self.modelo.predict(X_nuevo)
