# ML-Template 🧠

Proyecto personal de práctica de Data science y Machine Learning. Cuatro
módulos independientes, cada uno trabajando un tipo de problema distinto

## Estructura 📁

```
.
├── A.- Regression                  
│   ├── data/
│   ├── notebooks/
│   │   ├── 1.- EDA.ipynb
│   │   ├── 2.- seleccion.ipynb
│   │   └── 3.- busqueda.ipynb
│   ├── preprocessing/
│   └── src/
├── B.- Clasificacion               
│   ├── data/
│   ├── notebooks/
│   │   ├── 1.- EDA.ipynb
│   │   ├── 2.- Seleccion.ipynb
│   │   └── 3.- Busqueda.ipynb
│   ├── preprocessing/
│   └── src/
├── C.- Clustering                  
│   ├── data/
│   ├── notebooks/
│   │   ├── 1.- EDA.ipynb
│   │   ├── 2.- seleccion.ipynb
│   │   └── 3.- busqueda.ipynb
│   ├── preprocessing/
│   └── src/
├── D.- Deteccion-de-anomalias      
│   ├── notebooks/
│   │   ├── 1.- EDA.ipynb
│   │   ├── 2.- seleccion.ipynb
│   │   └── 3.- busqueda.ipynb
│   ├── preprocessing/
│   ├── scripts/
│   └── src/
├── README.md
└── requirements.txt
```

### A · Regresión 🏠

Predicción del valor mediano de viviendas en Boston.

### B · Clasificación 🩺

Clasificación binaria de tumores de mama como malignos o benignos.

### C · Clustering 🛒

Análisis exploratorio de segmentación de clientes de un centro comercial.

### D · Detección de anomalías 💳

Detección de transacciones fraudulentas con tarjetas de crédito sobre un
dataset altamente desbalanceado (0.17% de fraudes). Incluye script para
descargar el dataset desde Kaggle y generar una muestra balanceada.

## Setup ⚙️

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso 🚀

Cada módulo sigue el mismo flujo de 3 notebooks:

1. **EDA** 🔍 — Análisis exploratorio de datos
2. **seleccion** 🧹 — Preprocesamiento y selección de características
3. **busqueda** 🎯 — Búsqueda de hiperparámetros, entrenamiento y evaluación

Además, `preprocessing/` y `src/` contienen scripts reutilizables de
limpieza, división de datos y configuración.