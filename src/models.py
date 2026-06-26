"""
models.py
---------
Define la jerarquia orientada a objetos del corpus biblico:

    Biblia -> Testamento -> Libro -> Capitulo -> Versiculo

Cada clase usa __init__ explicito (sin dataclasses) para mantener el
mismo estilo usado en otros laboratorios del curso. El Libro ademas
guarda su genero (Law, Prophecy, Gospel, etc.), obtenido del archivo
key_genre_english.csv del dataset.
"""

import pandas as pd


class Versiculo:
    """
    Clase que representa un versiculo (unidad minima de analisis).

    Attributes:
        libro (str): Nombre del libro al que pertenece.
        capitulo (int): Numero de capitulo.
        numero (int): Numero de versiculo dentro del capitulo.
        texto_original (str): Texto del versiculo sin procesar.
        texto_procesado (list[str]): Tokens del versiculo tras el pipeline
            de preprocesamiento (vacio hasta que se procese).
    """

    def __init__(self, libro, capitulo, numero, texto_original):
        self.libro = libro
        self.capitulo = capitulo
        self.numero = numero
        self.texto_original = texto_original
        self.texto_procesado = []

    def set_texto_procesado(self, tokens):
        """
        Asigna los tokens resultantes del preprocesamiento.
        Args:
            tokens (list[str]): Lista de tokens ya procesados.
        """
        self.texto_procesado = tokens

    def get_texto_limpio(self):
        """
        Reconstruye el texto procesado como string (util para TF-IDF).
        Returns:
            str: Texto procesado unido por espacios.
        """
        return " ".join(self.texto_procesado)

    def __str__(self):
        return f"{self.libro} {self.capitulo}:{self.numero} - {self.texto_original[:40]}..."


class Capitulo:
    """
    Clase que representa un capitulo de un libro.

    Attributes:
        numero (int): Numero del capitulo.
        versiculos (list[Versiculo]): Versiculos que pertenecen al capitulo.
    """

    def __init__(self, numero):
        self.numero = numero
        self.versiculos = []

    def agregar_versiculo(self, versiculo):
        """
        Agrega un versiculo al capitulo.
        Args:
            versiculo (Versiculo): Versiculo por agregar.
        """
        self.versiculos.append(versiculo)

    def get_cantidad_versiculos(self):
        """
        Calcula la cantidad de versiculos del capitulo.
        Returns:
            int: Cantidad de versiculos.
        """
        return len(self.versiculos)

    def get_texto_completo(self):
        """
        Concatena el texto original de todos los versiculos del capitulo.
        Returns:
            str: Texto completo del capitulo.
        """
        return " ".join(v.texto_original for v in self.versiculos)

    def __str__(self):
        return f"Capitulo {self.numero} ({self.get_cantidad_versiculos()} versiculos)"


class Libro:
    """
    Clase que representa un libro de la Biblia.

    Attributes:
        nombre (str): Nombre del libro.
        testamento (str): Etiqueta del testamento, ej. "OT" o "NT".
        genero (str): Genero literario del libro, ej. "Law", "Gospel" (puede ser None
            si el dataset cargado no incluye esta informacion).
        capitulos (dict[int, Capitulo]): Capitulos del libro, indexados por numero.
    """

    def __init__(self, nombre, testamento, genero=None):
        self.nombre = nombre
        self.testamento = testamento
        self.genero = genero
        self.capitulos = {}

    def set_genero(self, genero):
        """
        Asigna o actualiza el genero literario del libro.
        Args:
            genero (str): Nombre del genero (ej. "Law", "Gospel").
        """
        self.genero = genero

    def get_genero(self):
        """
        Obtiene el genero literario del libro.
        Returns:
            str: Genero literario, o None si no fue asignado.
        """
        return self.genero

    def agregar_versiculo(self, versiculo):
        """
        Agrega un versiculo al libro, creando el capitulo si no existe.
        Args:
            versiculo (Versiculo): Versiculo por agregar.
        """
        if versiculo.capitulo not in self.capitulos:
            self.capitulos[versiculo.capitulo] = Capitulo(versiculo.capitulo)
        self.capitulos[versiculo.capitulo].agregar_versiculo(versiculo)

    def get_versiculos(self):
        """
        Obtiene todos los versiculos del libro (de todos sus capitulos).
        Returns:
            list[Versiculo]: Lista plana de versiculos.
        """
        out = []
        for cap in self.capitulos.values():
            out.extend(cap.versiculos)
        return out

    def get_cantidad_versiculos(self):
        """
        Calcula la cantidad total de versiculos del libro.
        Returns:
            int: Cantidad de versiculos.
        """
        return len(self.get_versiculos())

    def get_cantidad_capitulos(self):
        """
        Calcula la cantidad de capitulos del libro.
        Returns:
            int: Cantidad de capitulos.
        """
        return len(self.capitulos)

    def get_texto_completo(self):
        """
        Concatena el texto original de todos los versiculos del libro.
        Returns:
            str: Texto completo del libro.
        """
        return " ".join(v.texto_original for v in self.get_versiculos())

    def __str__(self):
        return (f"{self.nombre} (Testamento: {self.testamento}, Genero: {self.genero}, "
                f"{self.get_cantidad_versiculos()} versiculos)")


class Testamento:
    """
    Clase que representa un testamento (ej. "OT" o "NT").

    Attributes:
        nombre (str): Etiqueta del testamento.
        libros (dict[str, Libro]): Libros del testamento, indexados por nombre.
    """

    def __init__(self, nombre):
        self.nombre = nombre
        self.libros = {}

    def agregar_libro(self, libro):
        """
        Agrega un libro al testamento.
        Args:
            libro (Libro): Libro por agregar.
        """
        self.libros[libro.nombre] = libro

    def get_versiculos(self):
        """
        Obtiene todos los versiculos del testamento.
        Returns:
            list[Versiculo]: Lista plana de versiculos.
        """
        out = []
        for libro in self.libros.values():
            out.extend(libro.get_versiculos())
        return out

    def get_cantidad_versiculos(self):
        """
        Calcula la cantidad total de versiculos del testamento.
        Returns:
            int: Cantidad de versiculos.
        """
        return len(self.get_versiculos())

    def __str__(self):
        return f"Testamento {self.nombre} ({len(self.libros)} libros)"


class Biblia:
    """
    Clase raiz que representa el corpus completo.

    Los testamentos se crean dinamicamente segun las etiquetas que traiga
    el dataset (ej. "OT"/"NT" en key_english.csv), en vez de asumir nombres
    fijos como "Antiguo"/"Nuevo".

    Attributes:
        testamentos (dict[str, Testamento]): Testamentos encontrados en los datos,
            indexados por su etiqueta original.
    """

    def __init__(self):
        self.testamentos = {}

    def agregar_versiculo(self, versiculo, testamento, nombre_libro, genero=None):
        """
        Agrega un versiculo a la Biblia, creando testamento/libro si no existen.
        Args:
            versiculo (Versiculo): Versiculo por agregar.
            testamento (str): Etiqueta del testamento (ej. "OT", "NT").
            nombre_libro (str): Nombre del libro al que pertenece.
            genero (str, optional): Genero literario del libro (ej. "Law", "Gospel").
                Si el libro ya existe y no tenia genero asignado, se actualiza.
        """
        if testamento not in self.testamentos:
            self.testamentos[testamento] = Testamento(testamento)
        test_obj = self.testamentos[testamento]

        if nombre_libro not in test_obj.libros:
            test_obj.agregar_libro(Libro(nombre_libro, testamento, genero))
        elif genero is not None and test_obj.libros[nombre_libro].get_genero() is None:
            test_obj.libros[nombre_libro].set_genero(genero)

        test_obj.libros[nombre_libro].agregar_versiculo(versiculo)

    @staticmethod
    def from_dataframe(df, col_libro="libro", col_testamento="testamento",
                        col_capitulo="capitulo", col_versiculo="versiculo",
                        col_texto="texto", col_genero="genero"):
        """
        Construye una Biblia a partir de un DataFrame de pandas que tiene el siguiente formato:
        Index(['Verse ID', 'Book', 'Chapter', 'Verse', 'Text', 'Book Name',
       'Testament (OT or NT)', 'Genre ID', 'Genre name']
       (Los merges fueron hechos previamente)

        Args:
            df (pd.DataFrame): DataFrame con columnas de libro, testamento,
                capitulo, versiculo, texto y (opcionalmente) genero.
            col_libro (str): Nombre de la columna con el nombre del libro.
            col_testamento (str): Nombre de la columna con el testamento.
            col_capitulo (str): Nombre de la columna con el numero de capitulo.
            col_versiculo (str): Nombre de la columna con el numero de versiculo.
            col_texto (str): Nombre de la columna con el texto del versiculo.
            col_genero (str): Nombre de la columna con el genero literario del libro.
                Si no existe en el DataFrame, los libros quedan con genero=None.
        Returns:
            Biblia: Instancia construida con todos los versiculos del DataFrame.
        """
        biblia = Biblia()
        tiene_genero = col_genero in df.columns

        for _, row in df.iterrows():
            versiculo = Versiculo(
                libro=row[col_libro],
                capitulo=int(row[col_capitulo]),
                numero=int(row[col_versiculo]),
                texto_original=str(row[col_texto]),
            )
            genero = row[col_genero] if tiene_genero else None
            biblia.agregar_versiculo(
                versiculo,
                testamento=row[col_testamento],
                nombre_libro=row[col_libro],
                genero=genero,
            )
        return biblia

    def get_libros(self):
        """
        Obtiene todos los libros de la Biblia (todos los testamentos).
        Returns:
            list[Libro]: Lista de libros.
        """
        out = []
        for testamento in self.testamentos.values():
            out.extend(testamento.libros.values())
        return out

    def get_versiculos(self):
        """
        Obtiene todos los versiculos de la Biblia.
        Returns:
            list[Versiculo]: Lista plana de versiculos.
        """
        out = []
        for libro in self.get_libros():
            out.extend(libro.get_versiculos())
        return out

    def get_generos(self):
        """
        Obtiene la lista de generos literarios presentes en el corpus.
        Returns:
            list[str]: Generos unicos encontrados entre los libros (sin None).
        """
        generos = {libro.get_genero() for libro in self.get_libros()}
        generos.discard(None)
        return sorted(generos)

    def to_dataframe(self):
        """
        Aplana el corpus completo a un DataFrame de pandas.
        Returns:
            pd.DataFrame: Una fila por versiculo, con metadatos en el formato:
                Index(['testamento', 'libro', 'genero', 'capitulo', 'versiculo',
                'texto_original', 'texto_procesado', 'texto_limpio'],
                dtype='object')
        """
        rows = []
        for libro in self.get_libros():
            for versiculo in libro.get_versiculos():
                rows.append({
                    "testamento": libro.testamento,
                    "libro": libro.nombre,
                    "genero": libro.get_genero(),
                    "capitulo": versiculo.capitulo,
                    "versiculo": versiculo.numero,
                    "texto_original": versiculo.texto_original,
                    "texto_procesado": versiculo.texto_procesado,
                    "texto_limpio": versiculo.get_texto_limpio(),
                })
        return pd.DataFrame(rows)

    def get_resumen(self):
        """
        Calcula un resumen de cantidad de libros y versiculos por testamento.
        Returns:
            pd.DataFrame: Una fila por testamento con n_libros y n_versiculos.
        """
        rows = []
        for nombre, testamento in self.testamentos.items():
            rows.append({
                "testamento": nombre,
                "n_libros": len(testamento.libros),
                "n_versiculos": testamento.get_cantidad_versiculos(),
            })
        return pd.DataFrame(rows)

    def get_resumen_generos(self):
        """
        Calcula un resumen de cantidad de libros y versiculos por genero literario.
        Returns:
            pd.DataFrame: Una fila por genero con n_libros y n_versiculos.
        """
        rows = []
        for genero in self.get_generos():
            libros_genero = [l for l in self.get_libros() if l.get_genero() == genero]
            n_versiculos = sum(l.get_cantidad_versiculos() for l in libros_genero)
            rows.append({
                "genero": genero,
                "n_libros": len(libros_genero),
                "n_versiculos": n_versiculos,
            })
        return pd.DataFrame(rows)

    def __str__(self):
        return f"Biblia ({len(self.get_libros())} libros, {len(self.get_versiculos())} versiculos)"