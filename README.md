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
    class Biblia {
        +dict testamentos
        +from_dataframe(df) Biblia
        +agregar_versiculo(v, testamento, libro)
        +to_dataframe() DataFrame
        +resumen() DataFrame
    }
    class Testamento {
        +str nombre
        +dict libros
        +agregar_libro(libro)
    }
    class Libro {
        +str nombre
        +str testamento
        +dict capitulos
        +agregar_versiculo(v)
        +texto_completo
    }
    class Capitulo {
        +int numero
        +list versiculos
        +agregar_versiculo(v)
    }
    class Versiculo {
        +str libro
        +int capitulo
        +int numero
        +str texto_original
        +list texto_procesado
    }
    Biblia "1" *-- "2" Testamento
    Testamento "1" *-- "*" Libro
    Libro "1" *-- "*" Capitulo
    Capitulo "1" *-- "*" Versiculo

    class TextPreprocessor {
        +process(text) list
        +process_corpus(textos) list
        +palabras_mas_frecuentes(n)
    }

    class TFIDFVectorizer {
        +dict vocabulario
        +ndarray idf
        +fit(docs)
        +transform(docs) ndarray
        +fit_transform(docs) ndarray
    }

    class SemanticSearchEngine {
        +fit(df_corpus)
        +buscar(query, k) DataFrame
        +buscar_por_indice(idx, k) DataFrame
    }
    SemanticSearchEngine --> TFIDFVectorizer
    SemanticSearchEngine --> TextPreprocessor

    class VerseClassifier {
        +entrenar(X, y)
        +evaluar() dict
        +predecir(X) ndarray
    }

    class NGramModel {
        +int n
        +fit(oraciones)
        +generar(palabra_inicial, max_len) str
    }

    class SentimentAnalyzer {
        <<interface>>
        +score(texto) float
    }
    class LexiconSentimentAnalyzer
    class TextBlobSentimentAnalyzer
    SentimentAnalyzer <|-- TextBlobSentimentAnalyzer
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