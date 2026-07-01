# 📦Información sobre demanda, existencias y precios

Un panel de Streamlit que abarca tres casos de uso relacionados con el análisis del sector minorista: previsión de la demanda mediante series temporales, optimización de existencias y fijación dinámica de precios, basado en datos diarios de ventas y existencias a nivel de tienda y de producto.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)
![statsmodels](https://img.shields.io/badge/statsmodels-forecasting-informational)

## Características
 
### 📈 Previsión de la demanda
- Comparación entre la `previsión de la demanda` real y la comunicada para cualquier tienda o producto, con MAE, RMSE y MAPE.
- Un **modelo de suavizado exponencial de Holt-Winters** independiente, entrenado únicamente con datos históricos reales (no con la columna de previsiones declaradas), con una prueba retrospectiva de los últimos 30 días y una previsión prospectiva configurable (de 7 a 60 días) con un intervalo de confianza aproximado.
- Vistas de estacionalidad: demanda media por temporada y por día de la semana.

### 📦 Optimización de inventario
- **Plazo de entrega** y **nivel de servicio** configurables (barra lateral), que se utilizan para calcular el stock de seguridad y el punto de reposición por tienda/producto mediante fórmulas estándar de inventario.
- Cada combinación de tienda y producto se clasifica como **Riesgo de rotura de stock**, **Reposición inminente**, **Óptimo** o **Exceso de stock**, con una tabla de recomendaciones descargable.
- Rotación de inventario por categoría y un gráfico detallado que muestra el nivel de inventario a lo largo del tiempo en comparación con las líneas del punto de reposición y el stock de seguridad.

### 💰 Fijación dinámica de precios
- Una regresión log-log de la **elasticidad de la demanda respecto al precio** (teniendo en cuenta los descuentos y las fiestas/promociones), con el coeficiente, el valor p, el R² y una interpretación en lenguaje sencillo.
- Eficacia de los descuentos (media de unidades vendidas e ingresos medios por nivel de descuento).
- Análisis de la tendencia de los precios frente a los de la competencia y de las diferencias de precios, por categoría y por producto individual.

## Estructura del proyecto

```
.
├── app.py                       # Aplicación Streamlit
├── retail_store_inventory.csv   # Datos diarios de ventas e inventario de la tienda y los productos
└── README.md
```
## Primeros pasos

### Requisitos previos
- Python 3.9 o superior
- El archivo `retail_store_inventory.csv` en la misma carpeta que `app.py`

### Instalación
 
```bash
pip install streamlit pandas numpy plotly statsmodels
```

### Ejecutar la aplicación
 
```bash
streamlit run app.py
```

## Data
 
`retail_store_inventory.csv` — una fila por tienda/producto/día (5 tiendas × 20 productos × 731 días).
 
| Columna | Descripción |
|---|---|
| `Date`, `Store ID`, `Product ID` | Identificadores de fila |
| `Category`, `Region` | Atributos descriptivos (varían de un día a otro en este conjunto de datos; se tratan como dimensiones de filtro, no como propiedades fijas del producto o la tienda) |
| `Inventory Level` | Unidades en stock al inicio del día |
| `Units Sold`, `Units Ordered` | Ventas diarias y reposición |
| `Demand Forecast` | Un valor de previsión ya presente en los datos de origen, utilizado como referencia de comparación |
| `Price`, `Discount`, `Competitor Pricing` | Campos de precios |
| `Weather Condition`, `Holiday/Promotion`, `Seasonality` | Indicadores de contexto |
 
Columnas derivadas calculadas en el momento de la carga: `Revenue`, `Price_Gap` (precio propio − precio de la competencia), `Forecast_Error` (previsión comunicada − real).

## Notas metodológicas
 
- **Previsión**: Se aplica el modelo de Holt-Winters de forma independiente para cada tienda o selección de productos (tendencia aditiva + estacionalidad semanal), con una transición suave a un modelo no estacional si el ajuste estacional falla en una serie corta.
- **Inventario**: `Safety Stock = z × σ_demand × √(lead time)`, `Reorder Point = avg daily demand × lead time + Safety Stock`, utilizando la puntuación z de la distribución normal estándar para el nivel de servicio seleccionado. Este conjunto de datos repone el stock casi a diario (el inventario medio cubre solo ~1–2 días de demanda), por lo que el control deslizante del plazo de entrega se establece por defecto en un intervalo corto; si se ajusta a más de 7 días, la mayoría de los productos se marcarán como en riesgo, lo cual es de esperar dada la rápida cadencia de reposición de los datos, y no se trata de un error.
- **Precios**: la elasticidad se estima mediante OLS sobre el precio y la demanda transformados en logaritmos. En este conjunto de datos concreto (sintético), el precio y el descuento muestran una correlación muy débil con las unidades vendidas, por lo que el coeficiente de elasticidad a menudo resulta estadísticamente insignificante; el panel de control informa de ello con franqueza, en lugar de forzar una recomendación con un alto grado de confianza. La misma metodología producirá una estimación significativa de la elasticidad en datos con una relación real entre precio y demanda.

## Plataforma tecnológica

- [Streamlit](https://streamlit.io/) — marco de trabajo de la aplicación e interfaz de usuario
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — procesamiento de datos
- [Plotly](https://plotly.com/python/) — gráficos interactivos
- [statsmodels](https://www.statsmodels.org/) — previsión de Holt-Winters y regresión OLS

## Licencia
 
Este proyecto se ofrece tal cual, con fines educativos y para la creación de un portafolio.
