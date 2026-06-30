<a href="bibliabanner"><img src="./imgs/bible_banner.png" align="center" alt="bible" ></a>

<h1 align="center"> Taller 02 - Programación Científica </h1>

<p align = center>
<a href = "https://www.ucn.cl"><img alt="Static Badge" src="https://img.shields.io/badge/Universidad_Católica_del_Norte-orange"></a>
<a href = "https://eic.ucn.cl"> <img alt="Static Badge" src="https://img.shields.io/badge/Escuela_de_Ingeniería_Coquimbo-blue"></a>
</p>

## Biblical Text Mining — Laboratorio 2

El presente repositorio consiste en la implementación de un programa capaz de realizar análisis computacional de texto sobre el corpus bíblico, desarrollando labores de preprocesamiento, TF-IDF (implementado desde cero), motor de búsqueda
semántico, clasificador de versículos, generador de texto con n-gramas
y análisis de sentimiento.
Los datos para este taller fueron extraídos de https://www.kaggle.com/datasets/oswinrh/bible en la versión ASV (American Standard Version).

## Estructura del proyecto

```
biblical_text_mining/
├── data/                     
├── notebooks/            
├── src/
│   ├── __init__.py
│   ├── models.py 
|   ├── dataloader.py            
│   ├── preprocessing.py      
│   ├── tfidf.py              
│   ├── search_engine.py     
│   ├── classifier.py        
│   ├── ngram_model.py        
│   ├── sentiment.py          
│   └── visualization.py      
├── main.py                
├── requirements.txt
└── README.md
```

## Instalación

## Instalación
1. Clonar el repositorio y entrar en la carpeta:
``` bash
git clone https://github.com/Fifthtaschenmesser4/Taller02-ProgCien
```
2. Entrar en la raíz del proyecto:
```bash
cd Taller01-ProgCient
```
3. Descargar los requerimientos con el archivo __requirements.txt__:
``` bash
pip install -q -r requirements.txt
python main.py
```

## Diagrama de clases

```mermaid
classDiagram
direction LR

    class main {
        +main() DataFrame
    }

    class dataloader {
        +cargar_dataset(path_bible, path_key, path_genre) DataFrame
    }

    class visualization {
        +plot_longitud_versiculos(df, columna) Figure
        +plot_versiculos_por_libro(df) Figure
        +plot_heatmap_similitud_libros(matriz_similitud, nombres_libros) Figure
        +plot_pca_versiculos(matriz_tfidf, etiquetas, titulo) Figure
        +plot_sentimiento_por_libro(df_sentimiento_agregado) Figure
        +plot_wordcloud(frecuencias, titulo) Figure
        +plot_matriz_confusion(cm, clases) Figure
    }

    class Biblia {
        +dict testamentos
        +__init__()
        +agregar_versiculo(versiculo, testamento, nombre_libro, genero)
        +from_dataframe(df, col_libro, col_testamento, col_capitulo, col_versiculo, col_texto, col_genero) Biblia
        +get_libros() list
        +get_versiculos() list
        +get_generos() list
        +to_dataframe() DataFrame
        +get_resumen() DataFrame
        +get_resumen_generos() DataFrame
        +__str__() str
    }

    class Testamento {
        +str nombre
        +dict libros
        +__init__(nombre)
        +agregar_libro(libro)
        +get_versiculos() list
        +get_cantidad_versiculos() int
        +__str__() str
    }

    class Libro {
        +str nombre
        +str testamento
        +str genero
        +dict capitulos
        +__init__(nombre, testamento, genero)
        +set_genero(genero)
        +get_genero() str
        +agregar_versiculo(versiculo)
        +get_versiculos() list
        +get_cantidad_versiculos() int
        +get_cantidad_capitulos() int
        +get_texto_completo() str
        +__str__() str
    }

    class Capitulo {
        +int numero
        +list versiculos
        +__init__(numero)
        +agregar_versiculo(versiculo)
        +get_cantidad_versiculos() int
        +get_texto_completo() str
        +__str__() str
    }

    class Versiculo {
        +str libro
        +int capitulo
        +int numero
        +str texto_original
        +list texto_procesado
        +__init__(libro, capitulo, numero, texto_original)
        +set_texto_procesado(tokens)
        +get_texto_limpio() str
        +__str__() str
    }

    Biblia "1" *-- "*" Testamento
    Testamento "1" *-- "*" Libro
    Libro "1" *-- "*" Capitulo
    Capitulo "1" *-- "*" Versiculo

    class TextPreprocessor {
        +set stopwords
        +int min_token_len
        +bool lowercase
        +bool remove_punctuation
        +bool remove_numbers
        +bool remove_stopwords
        +dict vocabulario
        +Counter frecuencias
        +__init__(stopwords, min_token_len, lowercase, remove_punctuation, remove_numbers, remove_stopwords)
        +to_lowercase(text) str
        +strip_punctuation(text) str
        +strip_special_and_numbers(text) str
        +tokenize(text) list
        +filter_stopwords(tokens) list
        +filter_short_tokens(tokens) list
        +process(text) list
        +process_ngram(text) list
        +process_corpus(textos) list
        +process_corpus_ngram(textos) list
        +palabras_mas_frecuentes(n) list
    }

    class TFIDFVectorizer {
        +dict vocabulario
        +ndarray idf
        +int n_docs
        +bool normalizar
        +__init__(normalizar)
        +fit(documentos_tokenizados) TFIDFVectorizer
        +transform(documentos_tokenizados) ndarray
        +fit_transform(documentos_tokenizados) ndarray
        +vectorizar_texto_nuevo(tokens) ndarray
    }

    class tfidf {
        +cosine_similarity(vec_a, vec_b) float
        +cosine_similarity_matrix(matriz) ndarray
    }

    class SemanticSearchEngine {
        +TextPreprocessor preprocessor
        +TFIDFVectorizer vectorizer
        +ndarray matriz_tfidf
        +DataFrame df_corpus
        +__init__(preprocessor, vectorizer)
        +fit(df_corpus, columna_tokens) SemanticSearchEngine
        +buscar(query, k) DataFrame
        +buscar_por_indice(idx_versiculo, k) DataFrame
        -_rankear(vector_query, k) DataFrame
    }

    class VerseClassifier {
        +object modelo
        +str nombre_modelo
        +ndarray clases
        +ndarray X_test
        +Series y_test
        +__init__(modelo, maximo_iteraciones)
        +entrenar(X, y, test_size, random_state) tuple
        +evaluar() dict
        +predecir(X_nuevo) ndarray
        +__str__() str
    }

    class NGramModel {
        +int n
        +int contexto_size
        +dict conteos
        +set vocabulario
        +__init__(n)
        +fit(oraciones_tokenizadas) NGramModel
        +get_probabilidad(contexto, palabra) float
        +get_siguiente_palabra(contexto) str
        +generar(palabra_inicial, max_len) str
        +__str__() str
    }

    class ngram_model {
        +comparar_modelos(oraciones_tokenizadas, ns, palabra_inicial, max_len) dict
    }

    class SentimentAnalyzer {
        <<interface>>
        +score(texto) float
    }

    class TextBlobSentimentAnalyzer {
        +type text_blob_class
        +__init__()
        +score(texto) float
    }

    class sentiment {
        +calcular_sentimiento_corpus(df, analyzer, columna_texto) DataFrame
        +agregar_por_libro(df) DataFrame
        +agregar_por_capitulo(df) DataFrame
    }

    main ..> dataloader
    main ..> Biblia
    main ..> TextPreprocessor
    main ..> TFIDFVectorizer
    main ..> tfidf
    main ..> visualization
    main ..> SemanticSearchEngine
    main ..> VerseClassifier
    main ..> ngram_model
    main ..> sentiment

    SemanticSearchEngine --> TextPreprocessor
    SemanticSearchEngine --> TFIDFVectorizer
    SemanticSearchEngine ..> tfidf

    VerseClassifier --> TFIDFVectorizer
    ngram_model ..> NGramModel
    NGramModel --> TextPreprocessor

    TextBlobSentimentAnalyzer --|> SentimentAnalyzer
    sentiment ..> SentimentAnalyzer
```


## Integrantes
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/martindroguett">
        <img src="https://github.com/martindroguett.png" width="100px;" alt="Martín Droguett" style="border-radius:50%"/>
        <br />
        <sub><b>Martín Droguett Robledo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Fifthtaschenmesser4">
        <img src="https://github.com/Fifthtaschenmesser4.png" width="100px;" alt="Francisco Romero" style="border-radius:50%"/>
        <br />
        <sub><b>Francisco Romero Opazo</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/amelievalderrama-oss">
        <img src="https://github.com/amelievalderrama-oss.png" width="100px;" alt="Amelie Valderrama" style="border-radius:50%"/>
        <br />
        <sub><b>Amelie Valderrama</b></sub>
      </a>
    </td>
  </tr>
</table>