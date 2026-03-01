# 💬 appnlp — Análisis de Confesiones con NLP

Aplicación web interactiva construida con **Streamlit** para explorar y analizar un dataset de confesiones mediante técnicas de **Procesamiento de Lenguaje Natural (NLP)**. Permite buscar confesiones, filtrar por sentimiento y clúster, y visualizar estadísticas detalladas.

---

## 🚀 Características

- 🔍 **Buscador de confesiones**: búsqueda por palabras clave con filtros de fecha, sentimiento y clúster DBSCAN.
- 📊 **Estadísticas y análisis**:
  - Evolución de confesiones en el tiempo.
  - Top 20 usuarios con más comentarios.
  - Distribución de sentimientos en confesiones y comentarios (positivo, neutro, negativo).
- 💬 **Detalle de confesión**: texto original, comentarios con análisis de sentimiento y métricas agregadas.
- 🔗 **Confesiones similares**: similitud por coseno usando embeddings (BERT / TF-IDF).

---

## 🛠️ Tecnologías

| Categoría        | Herramienta              |
|------------------|--------------------------|
| Framework web    | [Streamlit](https://streamlit.io/) |
| Lenguaje         | Python 3.11              |
| Datos            | Pandas, NumPy            |
| Visualización    | Plotly                   |
| Archivos Excel   | openpyxl                 |

---

## 📁 Estructura del proyecto

```
appnlp/
├── app_streamlit.py               # Aplicación principal
├── requirements.txt               # Dependencias Python
├── comentarios_sentimientos.xlsx  # Comentarios con análisis de sentimiento
├── df_final_nlp5_part[1-16].csv  # Dataset de confesiones (16 partes)
└── splits_nlp5/                   # Directorio de splits adicionales
```

---

## ⚙️ Instalación

### Opción A: Entorno local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/paolosalazarp/appnlp.git
   cd appnlp
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Opción B: Dev Container (recomendado)

Abre el proyecto en [VS Code](https://code.visualstudio.com/) con la extensión **Dev Containers** instalada y selecciona **"Reopen in Container"**. Las dependencias se instalan automáticamente.

---

## ▶️ Uso

Ejecuta la aplicación con:

```bash
streamlit run app_streamlit.py
```

Luego abre tu navegador en [http://localhost:8501](http://localhost:8501).

---

## 🗂️ Datos esperados

Los archivos CSV deben contener las siguientes columnas (o equivalentes):

| Columna                   | Descripción                              |
|---------------------------|------------------------------------------|
| `texts`                   | Texto original de la confesión           |
| `comments`                | Lista de comentarios                     |
| `confe_limpio_sin_stem`   | Texto preprocesado sin stemming          |
| `sent_confe_etiqueta`     | Etiqueta de sentimiento (`POS`/`NEU`/`NEG`) |
| `sentimiento_puntaje`     | Puntaje numérico del sentimiento         |
| `date` / `fecha`          | Fecha de publicación                     |
| `users` / `user_name`     | Usuario autor                            |
| columnas `*embed*`        | Embeddings para similitud (BERT/TF-IDF)  |
| columnas `*dbscan*`       | Clústeres DBSCAN                         |

El archivo `comentarios_sentimientos.xlsx` debe incluir: `confesion_id`, `sentimiento_etiqueta`, `sentimiento_puntaje`.

---

## 📸 Vistas de la aplicación

| Vista               | Descripción                                      |
|---------------------|--------------------------------------------------|
| 🔍 Buscador         | Búsqueda y filtrado de confesiones               |
| 📊 Estadísticas     | Gráficos de evolución, usuarios y sentimientos   |

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos y de investigación en NLP.
