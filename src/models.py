"""
models.py
---------
Define la jerarquía orientada a objetos del corpus bíblico:

    Biblia -> Testamento -> Libro -> Capitulo -> Versiculo

Cada clase es responsable únicamente de su propio nivel y delega en sus
hijos cuando necesita agregar información (ej: contar versículos,
concatenar texto, etc). Esto facilita el diagrama de clases pedido en
los entregables.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class Versiculo:
    """Unidad mínima de análisis del corpus."""
    libro: str
    capitulo: int
    numero: int
    texto_original: str
    texto_procesado: List[str] = field(default_factory=list)  # tokens tras preprocesamiento

    def __repr__(self) -> str:
        return f"{self.libro} {self.capitulo}:{self.numero} - {self.texto_original[:40]}..."

    @property
    def texto_limpio(self) -> str:
        """Texto procesado reconstruido como string (útil para TF-IDF)."""
        return " ".join(self.texto_procesado)


@dataclass
class Capitulo:
    numero: int
    versiculos: List[Versiculo] = field(default_factory=list)

    def agregar_versiculo(self, versiculo: Versiculo) -> None:
        self.versiculos.append(versiculo)

    @property
    def cantidad_versiculos(self) -> int:
        return len(self.versiculos)

    @property
    def texto_completo(self) -> str:
        return " ".join(v.texto_original for v in self.versiculos)


@dataclass
class Libro:
    nombre: str
    testamento: str  # "Antiguo" | "Nuevo"
    capitulos: Dict[int, Capitulo] = field(default_factory=dict)

    def agregar_versiculo(self, versiculo: Versiculo) -> None:
        if versiculo.capitulo not in self.capitulos:
            self.capitulos[versiculo.capitulo] = Capitulo(numero=versiculo.capitulo)
        self.capitulos[versiculo.capitulo].agregar_versiculo(versiculo)

    @property
    def versiculos(self) -> List[Versiculo]:
        out = []
        for cap in self.capitulos.values():
            out.extend(cap.versiculos)
        return out

    @property
    def cantidad_versiculos(self) -> int:
        return len(self.versiculos)

    @property
    def cantidad_capitulos(self) -> int:
        return len(self.capitulos)

    @property
    def texto_completo(self) -> str:
        return " ".join(v.texto_original for v in self.versiculos)


class Testamento:
    def __init__(self, nombre: str):
        self.nombre = nombre  # "Antiguo" | "Nuevo"
        self.libros: Dict[str, Libro] = {}

    def agregar_libro(self, libro: Libro) -> None:
        self.libros[libro.nombre] = libro

    @property
    def versiculos(self) -> List[Versiculo]:
        out = []
        for libro in self.libros.values():
            out.extend(libro.versiculos)
        return out

    @property
    def cantidad_versiculos(self) -> int:
        return len(self.versiculos)


class Biblia:
    """
    Contenedor raíz del corpus. Se construye típicamente a partir de un
    DataFrame de pandas con columnas mínimas:
        ['libro', 'testamento', 'capitulo', 'versiculo', 'texto']
    Ajusta los nombres de columna en `from_dataframe` según el dataset
    de Kaggle que elijan (la estructura varía según la versión descargada).
    """

    def __init__(self):
        self.testamentos: Dict[str, Testamento] = {
            "Antiguo": Testamento("Antiguo"),
            "Nuevo": Testamento("Nuevo"),
        }

    def agregar_versiculo(self, versiculo: Versiculo, testamento: str, nombre_libro: str) -> None:
        test_obj = self.testamentos[testamento]
        if nombre_libro not in test_obj.libros:
            test_obj.agregar_libro(Libro(nombre=nombre_libro, testamento=testamento))
        test_obj.libros[nombre_libro].agregar_versiculo(versiculo)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        col_libro: str = "libro",
        col_testamento: str = "testamento",
        col_capitulo: str = "capitulo",
        col_versiculo: str = "versiculo",
        col_texto: str = "texto",
    ) -> "Biblia":
        biblia = cls()
        for _, row in df.iterrows():
            v = Versiculo(
                libro=row[col_libro],
                capitulo=int(row[col_capitulo]),
                numero=int(row[col_versiculo]),
                texto_original=str(row[col_texto]),
            )
            biblia.agregar_versiculo(v, testamento=row[col_testamento], nombre_libro=row[col_libro])
        return biblia

    @property
    def libros(self) -> List[Libro]:
        out = []
        for t in self.testamentos.values():
            out.extend(t.libros.values())
        return out

    @property
    def versiculos(self) -> List[Versiculo]:
        out = []
        for libro in self.libros:
            out.extend(libro.versiculos)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Aplana el corpus completo a un DataFrame, útil para el resto del pipeline."""
        rows = []
        for libro in self.libros:
            for v in libro.versiculos:
                rows.append({
                    "testamento": libro.testamento,
                    "libro": libro.nombre,
                    "capitulo": v.capitulo,
                    "versiculo": v.numero,
                    "texto_original": v.texto_original,
                    "texto_procesado": v.texto_procesado,
                    "texto_limpio": v.texto_limpio,
                })
        return pd.DataFrame(rows)

    def resumen(self) -> pd.DataFrame:
        """Cantidad de libros, capítulos y versículos por testamento (chequeo rápido)."""
        rows = []
        for nombre, t in self.testamentos.items():
            rows.append({
                "testamento": nombre,
                "n_libros": len(t.libros),
                "n_versiculos": t.cantidad_versiculos,
            })
        return pd.DataFrame(rows)
