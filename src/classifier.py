"""
classifier.py
--------------
Clasificador que predice el libro al que pertenece un versiculo, usando
como features la matriz TF-IDF (implementada a mano en tfidf.py) y como
modelo un clasificador estandar de sklearn (esto SI esta permitido, la
restriccion del enunciado es solo sobre TF-IDF y similitud coseno).
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class VerseClassifier:
    """
    Clase que representa un clasificador de versiculos por libro.

    Attributes:
        modelo (object): Instancia del modelo de sklearn (LogisticRegression
            o MultinomialNB) usado internamente.
        nombre_modelo (str): Nombre del modelo elegido ("logistic" o "naive_bayes").
        clases (np.ndarray): Lista de libros (etiquetas) vistos durante el entrenamiento.
        X_test (np.ndarray): Features de prueba, guardadas tras entrenar() para poder evaluar().
        y_test (pd.Series): Etiquetas reales de prueba, guardadas tras entrenar().
    """

    def __init__(self, modelo="logistic"):
        """
        Inicializa el clasificador segun el tipo de modelo elegido.
        Args:
            modelo (str): "logistic" para regresion logistica, o "naive_bayes"
                para Naive Bayes Multinomial (baseline simple y rapido).
        """
        if modelo == "logistic":
            self.modelo = LogisticRegression(max_iter=1000)
        elif modelo == "naive_bayes":
            self.modelo = MultinomialNB()
        else:
            raise ValueError("modelo debe ser 'logistic' o 'naive_bayes'")

        self.nombre_modelo = modelo
        self.clases = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, X, y, test_size=0.2, random_state=42):
        """
        Separa los datos en entrenamiento/prueba y entrena el modelo.
        Args:
            X (np.ndarray): Matriz de features (ej. TF-IDF de los versiculos).
            y (pd.Series): Etiquetas (libro al que pertenece cada versiculo).
            test_size (float): Proporcion de datos reservados para prueba.
            random_state (int): Semilla para la separacion train/test.
        Returns:
            tuple: (X_train, X_test, y_train, y_test) usados internamente.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.modelo.fit(X_train, y_train)
        self.clases = self.modelo.classes_
        self.X_test = X_test
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def evaluar(self):
        """
        Evalua el modelo sobre el conjunto de prueba guardado en entrenar().
        Returns:
            dict: Contiene accuracy, matriz_confusion, clases, reporte,
                y_test e y_pred.
        """
        if self.X_test is None:
            raise RuntimeError("Llama a entrenar() antes de evaluar().")

        y_pred = self.modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matriz_confusion = confusion_matrix(self.y_test, y_pred, labels=self.clases)
        reporte = classification_report(self.y_test, y_pred, labels=self.clases, zero_division=0)
        return {
            "accuracy": accuracy,
            "matriz_confusion": matriz_confusion,
            "clases": self.clases,
            "reporte": reporte,
            "y_test": self.y_test,
            "y_pred": y_pred,
        }

    def predecir(self, X_nuevo):
        """
        Predice el libro para nuevas filas de features.
        Args:
            X_nuevo (np.ndarray): Matriz de features de los versiculos a predecir.
        Returns:
            np.ndarray: Libros predichos para cada fila de X_nuevo.
        """
        return self.modelo.predict(X_nuevo)

    def __str__(self):
        return f"VerseClassifier(modelo={self.nombre_modelo}, clases={self.clases})"