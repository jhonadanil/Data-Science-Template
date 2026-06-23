# ML-Template

Proyecto personal de práctica de Data science y Machine Learning. Contiene cuatro módulos independientes, cada uno trabajando un tipo de problema distinto
con datasets clásicos.

### A · Regresión

Predicción del valor mediano de viviendas en Bosto

### B · Clasificación

Clasificación binaria de tumores de mama como malignos o benignos 

### C · Clustering

Análisis exploratorio de segmentación de clientes de un centro
comercial 

### D · Detección de anomalías

Detección de transacciones fraudulentas con tarjetas de crédito sobre un dataset altamente desbalanceado (0.17%
de fraudes). Incluye script para descargar el dataset desde Kaggle y
generar una muestra balanceada.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Cada módulo tiene notebooks en su carpeta `notebooks/` que documentan
el flujo completo (EDA, feature selection, modelado, evaluación) y
scripts reutilizables de division de datos, preprocesamiento, limpieza y configuracion para un futuro src.