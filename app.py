import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# 1. CONFIGURACION DE LA PAGINA
st.set_page_config(
    page_title="Gestión de la demanda, el inventario y los precios",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_COLOR = "#0B111E"
CARD_COLOR = "#141B2D"
BORDER_COLOR = "#232F4B"
ACCENT = "#00E5FF"
ACCENT_2 = "#7C5CFF"
GOOD = "#34D399"
WARN = "#FBBF24"
BAD = "#FF5C7A"
TEXT_MUTED = "#8892B0"

PLOTLY_TEMPLATE = "plotly_dark"
STATUS_COLORS = {
    "Stockout Risk": BAD,
    "Reorder Soon": WARN,
    "Optimal": GOOD,
    "Overstock": ACCENT_2,
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG_COLOR}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetricValue"] {{ color: {ACCENT}; font-size: 24px; font-weight: 700; }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
    div[data-testid="stMetricDelta"] {{ color: {GOOD}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD_COLOR};
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
    }}
    h1, h2, h3 {{ color: #ffffff; }}
    .subtitle {{ color: {TEXT_MUTED}; font-size: 15px; margin-top: -10px; }}
    .section-title {{ color: #ffffff; font-weight: 600; margin-bottom: 4px; }}
    .insight-box {{
        background-color: rgba(0, 229, 255, 0.08);
        border-left: 3px solid {ACCENT};
        padding: 12px 16px;
        border-radius: 6px;
        color: #dbe2f0;
        font-size: 14.5px;
    }}
    </style>
    """, unsafe_allow_html=True
)

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e9f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

Z_SCORES = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}

# CARGA DE DATOS
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["Revenue"] = df["Units Sold"] * df["Price"] * (1 - df["Discount"] / 100)
    df["Price_Gap"] = df["Price"] - df["Competitor Pricing"]
    df["Forecast_Error"] = df["Demand Forecast"] - df["Units Sold"]
    return df
 
 
try:
    raw_df = load_data()
except FileNotFoundError:
    st.error("No se encontro 'retail_store_inventory.csv'. Colocalo junto a app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()


# ======================================================================================
# 3. BARRA LATERAL — FILTROS GLOBALES
# ======================================================================================
st.sidebar.title("🎛️ Panel de Control")
st.sidebar.markdown("Filtra los datos analizados en las tres pestanas.")
 
tiendas = sorted(raw_df["Store ID"].unique())
categorias = sorted(raw_df["Category"].unique())
regiones = sorted(raw_df["Region"].unique())
 
tiendas_sel = st.sidebar.multiselect("Tienda", tiendas, default=tiendas)
categorias_sel = st.sidebar.multiselect("Categoria", categorias, default=categorias)
regiones_sel = st.sidebar.multiselect("Region", regiones, default=regiones)
 
fecha_min = raw_df["Date"].min().to_pydatetime()
fecha_max = raw_df["Date"].max().to_pydatetime()
fechas_sel = st.sidebar.slider(
    "Rango de Fechas", min_value=fecha_min, max_value=fecha_max, value=(fecha_min, fecha_max), format="DD-MM-YYYY"
)
 
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Supuestos de Inventario")
lead_time = st.sidebar.slider("Lead Time del Proveedor (dias)", 1, 14, 1)
nivel_servicio = st.sidebar.select_slider("Nivel de Servicio Objetivo", options=list(Z_SCORES.keys()), value="95%")
st.sidebar.caption(
    "Usado para calcular stock de seguridad y punto de reorden en la pestana de Inventario. "
    "Nota: en este dataset el inventario se repone casi a diario (cobertura promedio ≈1-2 dias de demanda), "
    "por lo que un lead time de 1-3 dias refleja mejor el ciclo real de reabastecimiento."
)
 
df = raw_df[
    (raw_df["Store ID"].isin(tiendas_sel))
    & (raw_df["Category"].isin(categorias_sel))
    & (raw_df["Region"].isin(regiones_sel))
    & (raw_df["Date"] >= fechas_sel[0])
    & (raw_df["Date"] <= fechas_sel[1])
].copy()
 
st.sidebar.markdown("---")
st.sidebar.caption(f"📄 {len(raw_df):,} registros historicos | {raw_df['Store ID'].nunique()} tiendas | {raw_df['Product ID'].nunique()} productos")
 
# ======================================================================================
# 4. ENCABEZADO + KPIs GLOBALES
# ======================================================================================
st.title("📦 Gestión de la demanda, el inventario y los precios")
st.markdown(
    '<p class="subtitle">Pronostico de demanda, optimizacion de inventario y estrategia de precios dinamicos '
    "para una cadena minorista multi-tienda.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")
 
if df.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()
 
total_unidades = df["Units Sold"].sum()
ingresos_totales = df["Revenue"].sum()
inv_promedio = df["Inventory Level"].mean()
rotacion = total_unidades / inv_promedio if inv_promedio else 0
mape_forecast = (df["Forecast_Error"].abs() / df["Units Sold"].replace(0, np.nan)).mean() * 100
 
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unidades Vendidas", f"{total_unidades:,.0f}")
k2.metric("Ingresos", f"${ingresos_totales:,.0f}")
k3.metric("Inventario Promedio", f"{inv_promedio:,.0f} u.")
k4.metric("Rotacion de Inventario", f"{rotacion:.2f}x")
k5.metric("MAPE Forecast Reportado", f"{mape_forecast:.1f}%")
 
st.markdown("")
 
tab1, tab2, tab3 = st.tabs(["📈 Pronostico de Demanda", "📦 Optimizacion de Inventario", "💰 Precios Dinamicos"])
 
# ======================================================================================
# TAB 1 — PRONOSTICO DE DEMANDA
# ======================================================================================
with tab1:
    st.subheader("Pronostico de Demanda por Producto y Tienda")
    st.markdown("Selecciona una combinacion tienda-producto para analizar su serie temporal y proyectar su demanda futura.")
 
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        tienda_fc = st.selectbox("Tienda", sorted(df["Store ID"].unique()), key="fc_store")
    productos_disponibles = sorted(df[df["Store ID"] == tienda_fc]["Product ID"].unique())
    with col_sel2:
        producto_fc = st.selectbox("Producto", productos_disponibles, key="fc_product")
    with col_sel3:
        horizonte = st.slider("Horizonte de pronostico (dias)", 7, 60, 30)
 
    serie = (
        raw_df[(raw_df["Store ID"] == tienda_fc) & (raw_df["Product ID"] == producto_fc)]
        .set_index("Date")[["Units Sold", "Demand Forecast"]]
        .asfreq("D")
    )
    serie["Units Sold"] = serie["Units Sold"].interpolate()
    serie["Demand Forecast"] = serie["Demand Forecast"].interpolate()
 
    with st.container(border=True):
        st.markdown('<p class="section-title">Demanda Historica: Real vs. Forecast Reportado</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=serie.index, y=serie["Units Sold"], name="Unidades Vendidas (Real)", line=dict(color=ACCENT)))
        fig_hist.add_trace(go.Scatter(x=serie.index, y=serie["Demand Forecast"], name="Forecast Reportado", line=dict(color=ACCENT_2, dash="dot")))
        fig_hist.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Fecha", yaxis_title="Unidades")
        st.plotly_chart(style_fig(fig_hist), width="stretch")
 
        mae_r = serie["Forecast_Error"].abs().mean() if "Forecast_Error" in serie else (serie["Demand Forecast"] - serie["Units Sold"]).abs().mean()
        rmse_r = np.sqrt(((serie["Demand Forecast"] - serie["Units Sold"]) ** 2).mean())
        mape_r = ((serie["Demand Forecast"] - serie["Units Sold"]).abs() / serie["Units Sold"].replace(0, np.nan)).mean() * 100
        mr1, mr2, mr3 = st.columns(3)
        mr1.metric("MAE (forecast reportado)", f"{mae_r:.1f} u.")
        mr2.metric("RMSE (forecast reportado)", f"{rmse_r:.1f} u.")
        mr3.metric("MAPE (forecast reportado)", f"{mape_r:.1f}%")
 
    # ---- Modelo estadistico propio (Holt-Winters) ----
    with st.container(border=True):
        st.markdown('<p class="section-title">Modelo Estadistico: Suavizado Exponencial (Holt-Winters)</p>', unsafe_allow_html=True)
        st.caption(
            "Este modelo se entrena unicamente con el historico de unidades vendidas (sin usar la columna "
            "'Demand Forecast'), simulando un escenario real donde se debe predecir la demanda desde cero."
        )
 
        y = serie["Units Sold"]
        backtest_days = min(30, len(y) // 4)
        train, test = y.iloc[:-backtest_days], y.iloc[-backtest_days:]
 
        def fit_forecast(train_series, periods):
            try:
                model = ExponentialSmoothing(
                    train_series, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated"
                ).fit()
                return model.forecast(periods)
            except Exception:
                # Fallback: suavizado exponencial simple sin estacionalidad
                model = ExponentialSmoothing(train_series, trend="add", initialization_method="estimated").fit()
                return model.forecast(periods)
 
        fc_backtest = fit_forecast(train, backtest_days)
        mae_model = np.abs(fc_backtest.values - test.values).mean()
        rmse_model = np.sqrt(((fc_backtest.values - test.values) ** 2).mean())
        mape_model = (np.abs(fc_backtest.values - test.values) / np.where(test.values == 0, np.nan, test.values)).mean() * 100 if not test.eq(0).all() else np.nan
 
        fc_future = fit_forecast(y, horizonte)
        resid_std = np.abs(fc_backtest.values - test.values).std()
 
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=y.index[-90:], y=y.values[-90:], name="Historico (Real)", line=dict(color=ACCENT)))
        fig_fc.add_trace(go.Scatter(x=fc_future.index, y=fc_future.values, name="Pronostico", line=dict(color=GOOD, dash="dash")))
        fig_fc.add_trace(
            go.Scatter(
                x=list(fc_future.index) + list(fc_future.index[::-1]),
                y=list(fc_future.values + 1.28 * resid_std) + list((fc_future.values - 1.28 * resid_std)[::-1]),
                fill="toself",
                fillcolor="rgba(52, 211, 153, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Banda de confianza (~80%)",
                showlegend=True,
            )
        )
        fig_fc.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Fecha", yaxis_title="Unidades")
        st.plotly_chart(style_fig(fig_fc), width="stretch")
 
        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("MAE del modelo (backtest)", f"{mae_model:.1f} u.")
        bm2.metric("RMSE del modelo (backtest)", f"{rmse_model:.1f} u.")
        bm3.metric("MAPE del modelo (backtest)", f"{mape_model:.1f}%" if not np.isnan(mape_model) else "N/D")
 
        st.markdown(
            f"""<div class="insight-box">
            📌 El forecast reportado en el dataset tiene un MAE de {mae_r:.1f} unidades, mientras que un modelo
            estadistico entrenado solo con el historico (sin conocer el forecast reportado) logra un MAE de
            {mae_model:.1f} en el mismo horizonte de prueba. Esto ilustra la diferencia entre un forecast ya
            calibrado por el negocio y uno construido puramente desde series de tiempo — util para decidir si
            vale la pena invertir en un modelo mas sofisticado (ej. regresores externos como clima o promociones).
            </div>""",
            unsafe_allow_html=True,
        )
 
    # ---- Estacionalidad ----
    with st.container(border=True):
        st.markdown('<p class="section-title">Patrones de Estacionalidad</p>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            por_temporada = df.groupby("Seasonality")["Units Sold"].mean().reindex(["Spring", "Summer", "Autumn", "Winter"]).reset_index()
            fig_temp = px.bar(
                por_temporada, x="Seasonality", y="Units Sold", template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[ACCENT], labels={"Units Sold": "Unidades promedio", "Seasonality": "Temporada"},
            )
            st.plotly_chart(style_fig(fig_temp), width="stretch")
        with col_s2:
            df_dow = df.copy()
            df_dow["Dia_Semana"] = df_dow["Date"].dt.day_name()
            orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            por_dia = df_dow.groupby("Dia_Semana")["Units Sold"].mean().reindex(orden_dias).reset_index()
            fig_dow = px.bar(
                por_dia, x="Dia_Semana", y="Units Sold", template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[ACCENT_2], labels={"Units Sold": "Unidades promedio", "Dia_Semana": ""},
            )
            st.plotly_chart(style_fig(fig_dow), width="stretch")
 
# ======================================================================================
# TAB 2 — OPTIMIZACION DE INVENTARIO
# ======================================================================================
with tab2:
    st.subheader("Optimizacion de Niveles de Inventario")
    st.markdown(
        f"Calculado con **lead time = {lead_time} dias** y **nivel de servicio = {nivel_servicio}** "
        "(ajustables en la barra lateral)."
    )
 
    z = Z_SCORES[nivel_servicio]
 
    @st.cache_data
    def calcular_inventario(data, lead_time, z):
        agg = data.groupby(["Store ID", "Product ID"]).agg(
            Demanda_Promedio=("Units Sold", "mean"),
            Demanda_Std=("Units Sold", "std"),
            Inventario_Actual=("Inventory Level", "last"),
            Categoria=("Category", "last"),
        ).reset_index()
        agg["Demanda_Std"] = agg["Demanda_Std"].fillna(0)
        agg["Stock_Seguridad"] = z * agg["Demanda_Std"] * np.sqrt(lead_time)
        agg["Punto_Reorden"] = agg["Demanda_Promedio"] * lead_time + agg["Stock_Seguridad"]
        agg["Dias_de_Cobertura"] = agg["Inventario_Actual"] / agg["Demanda_Promedio"].replace(0, np.nan)
 
        # Umbral de sobre-stock simplificado: mas del doble del punto de reorden en unidades
        agg["Estado"] = np.select(
            [
                agg["Inventario_Actual"] <= agg["Stock_Seguridad"],
                agg["Inventario_Actual"] <= agg["Punto_Reorden"],
                agg["Inventario_Actual"] > 1.5 * agg["Punto_Reorden"],
            ],
            ["Stockout Risk", "Reorder Soon", "Overstock"],
            default="Optimal",
        )
        return agg
 
    inv_df = calcular_inventario(df, lead_time, z)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Riesgo de Quiebre", int((inv_df["Estado"] == "Stockout Risk").sum()))
    c2.metric("🟡 Reordenar Pronto", int((inv_df["Estado"] == "Reorder Soon").sum()))
    c3.metric("🟢 Nivel Optimo", int((inv_df["Estado"] == "Optimal").sum()))
    c4.metric("🔵 Sobre-stock", int((inv_df["Estado"] == "Overstock").sum()))
 
    st.markdown("")
    col_dist, col_cat = st.columns([1, 1])
    with col_dist:
        with st.container(border=True):
            st.markdown('<p class="section-title">Distribucion de Estados de Inventario</p>', unsafe_allow_html=True)
            estado_counts = inv_df["Estado"].value_counts().reset_index()
            estado_counts.columns = ["Estado", "Cantidad"]
            fig_estado = px.pie(
                estado_counts, names="Estado", values="Cantidad", hole=0.55,
                color="Estado", color_discrete_map=STATUS_COLORS, template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(style_fig(fig_estado), width="stretch")
    with col_cat:
        with st.container(border=True):
            st.markdown('<p class="section-title">Rotacion de Inventario por Categoria</p>', unsafe_allow_html=True)
            rot_cat = df.groupby("Category").apply(
                lambda x: x["Units Sold"].sum() / x["Inventory Level"].mean() if x["Inventory Level"].mean() else 0
            ).reset_index(name="Rotacion")
            fig_rot = px.bar(
                rot_cat.sort_values("Rotacion", ascending=False), x="Category", y="Rotacion",
                template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT],
                labels={"Rotacion": "Unidades vendidas / Inventario promedio", "Category": ""},
            )
            st.plotly_chart(style_fig(fig_rot), width="stretch")
 
    st.markdown("")
    with st.container(border=True):
        st.markdown('<p class="section-title">📋 Tabla de Recomendaciones de Reabastecimiento</p>', unsafe_allow_html=True)
        filtro_estado = st.multiselect(
            "Filtrar por estado", list(STATUS_COLORS.keys()), default=list(STATUS_COLORS.keys())
        )
        tabla = inv_df[inv_df["Estado"].isin(filtro_estado)].copy()
        tabla = tabla.round(1).sort_values("Estado")
        st.dataframe(
            tabla[["Store ID", "Product ID", "Categoria", "Inventario_Actual", "Demanda_Promedio", "Stock_Seguridad", "Punto_Reorden", "Dias_de_Cobertura", "Estado"]],
            width="stretch",
            column_config={"Estado": st.column_config.TextColumn("Estado")},
        )
        st.download_button(
            "⬇️ Descargar recomendaciones (CSV)", data=tabla.to_csv(index=False).encode("utf-8"),
            file_name="recomendaciones_inventario.csv", mime="text/csv",
        )
 
    with st.container(border=True):
        st.markdown('<p class="section-title">Detalle: Inventario vs. Puntos de Control</p>', unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            tienda_inv = st.selectbox("Tienda", sorted(df["Store ID"].unique()), key="inv_store")
        with col_d2:
            producto_inv = st.selectbox("Producto", sorted(df[df["Store ID"] == tienda_inv]["Product ID"].unique()), key="inv_product")
 
        fila = inv_df[(inv_df["Store ID"] == tienda_inv) & (inv_df["Product ID"] == producto_inv)].iloc[0]
        serie_inv = raw_df[(raw_df["Store ID"] == tienda_inv) & (raw_df["Product ID"] == producto_inv)].sort_values("Date")
 
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(x=serie_inv["Date"], y=serie_inv["Inventory Level"], name="Inventario", line=dict(color=ACCENT)))
        fig_inv.add_hline(y=fila["Punto_Reorden"], line_dash="dash", line_color=WARN, annotation_text="Punto de Reorden")
        fig_inv.add_hline(y=fila["Stock_Seguridad"], line_dash="dash", line_color=BAD, annotation_text="Stock de Seguridad")
        fig_inv.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Fecha", yaxis_title="Unidades en Inventario")
        st.plotly_chart(style_fig(fig_inv), width="stretch")
 
# ======================================================================================
# TAB 3 — PRECIOS DINAMICOS
# ======================================================================================
with tab3:
    st.subheader("Estrategia de Precios Dinamicos")
    st.markdown("Analiza la relacion entre precio, descuentos, precio de la competencia y demanda.")
 
    # ---- Elasticidad precio-demanda (regresion log-log) ----
    with st.container(border=True):
        st.markdown('<p class="section-title">Elasticidad Precio-Demanda</p>', unsafe_allow_html=True)
        st.caption("Regresion log-log: log(Unidades Vendidas + 1) ~ log(Precio) + Descuento + Promocion/Feriado")
 
        reg_df = df[["Units Sold", "Price", "Discount", "Holiday/Promotion"]].copy()
        reg_df = reg_df[(reg_df["Price"] > 0)]
        reg_df["log_units"] = np.log1p(reg_df["Units Sold"])
        reg_df["log_price"] = np.log(reg_df["Price"])
 
        X = sm.add_constant(reg_df[["log_price", "Discount", "Holiday/Promotion"]])
        y_reg = reg_df["log_units"]
        modelo_elasticidad = sm.OLS(y_reg, X).fit()
 
        elasticidad = modelo_elasticidad.params["log_price"]
        p_valor = modelo_elasticidad.pvalues["log_price"]
        r2 = modelo_elasticidad.rsquared
 
        e1, e2, e3 = st.columns(3)
        e1.metric("Elasticidad Precio", f"{elasticidad:.3f}")
        e2.metric("p-valor", f"{p_valor:.3f}")
        e3.metric("R² del modelo", f"{r2:.3f}")
 
        if p_valor > 0.05:
            interpretacion = (
                "El coeficiente de precio **no es estadisticamente significativo** (p > 0.05) en los datos "
                "filtrados. En este dataset, el precio no parece ser un impulsor fuerte de la demanda — las "
                "variaciones en unidades vendidas se explican mas por otros factores (estacionalidad, "
                "inventario disponible, etc.). Se recomienda validar con datos reales de mercado antes de "
                "usar esta elasticidad para fijar precios."
            )
        elif elasticidad < -1:
            interpretacion = (
                f"Demanda **elastica** (elasticidad = {elasticidad:.2f}): un aumento de precio reduce las "
                "unidades vendidas mas que proporcionalmente. Bajar el precio podria aumentar el ingreso total."
            )
        elif elasticidad < 0:
            interpretacion = (
                f"Demanda **inelastica** (elasticidad = {elasticidad:.2f}): la demanda cae menos que "
                "proporcionalmente ante subidas de precio. Existe margen para subir precios sin perder muchas unidades."
            )
        else:
            interpretacion = (
                f"Coeficiente de precio positivo ({elasticidad:.2f}), un resultado atipico que sugiere que otras "
                "variables (calidad percibida, categoria) dominan la relacion precio-demanda en este subconjunto."
            )
        st.markdown(f'<div class="insight-box">📌 {interpretacion}</div>', unsafe_allow_html=True)
 
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        with st.container(border=True):
            st.markdown('<p class="section-title">Efectividad del Descuento</p>', unsafe_allow_html=True)
            disc_group = df.groupby("Discount").agg(Unidades_Promedio=("Units Sold", "mean"), Ingreso_Promedio=("Revenue", "mean")).reset_index()
            fig_disc = go.Figure()
            fig_disc.add_trace(go.Bar(x=disc_group["Discount"], y=disc_group["Unidades_Promedio"], name="Unidades Promedio", marker_color=ACCENT))
            fig_disc.add_trace(go.Scatter(x=disc_group["Discount"], y=disc_group["Ingreso_Promedio"], name="Ingreso Promedio", yaxis="y2", line=dict(color=GOOD)))
            fig_disc.update_layout(
                template=PLOTLY_TEMPLATE, xaxis_title="Descuento (%)",
                yaxis=dict(title="Unidades Promedio"),
                yaxis2=dict(title="Ingreso Promedio (USD)", overlaying="y", side="right"),
            )
            st.plotly_chart(style_fig(fig_disc), width="stretch")
 
    with col_p2:
        with st.container(border=True):
            st.markdown('<p class="section-title">Brecha de Precio vs. Competencia</p>', unsafe_allow_html=True)
            gap_cat = df.groupby("Category")["Price_Gap"].mean().reset_index().sort_values("Price_Gap")
            fig_gap = px.bar(
                gap_cat, x="Price_Gap", y="Category", orientation="h", template=PLOTLY_TEMPLATE,
                color="Price_Gap", color_continuous_scale=[BAD, "#232f4b", GOOD],
                labels={"Price_Gap": "Precio propio - Precio competencia (USD)", "Category": ""},
            )
            fig_gap.add_vline(x=0, line_dash="dash", line_color=TEXT_MUTED)
            st.plotly_chart(style_fig(fig_gap), width="stretch")
 
    with st.container(border=True):
        st.markdown('<p class="section-title">Precio Propio vs. Competencia (detalle por producto)</p>', unsafe_allow_html=True)
        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            tienda_p = st.selectbox("Tienda", sorted(df["Store ID"].unique()), key="price_store")
        with col_pp2:
            producto_p = st.selectbox("Producto", sorted(df[df["Store ID"] == tienda_p]["Product ID"].unique()), key="price_product")
 
        serie_p = raw_df[(raw_df["Store ID"] == tienda_p) & (raw_df["Product ID"] == producto_p)].sort_values("Date")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=serie_p["Date"], y=serie_p["Price"], name="Precio Propio", line=dict(color=ACCENT)))
        fig_price.add_trace(go.Scatter(x=serie_p["Date"], y=serie_p["Competitor Pricing"], name="Precio Competencia", line=dict(color=ACCENT_2, dash="dot")))
        fig_price.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Fecha", yaxis_title="Precio (USD)")
        st.plotly_chart(style_fig(fig_price), width="stretch")
 
        gap_actual = serie_p["Price_Gap"].iloc[-1]
        corr_precio_unidades = serie_p["Price"].corr(serie_p["Units Sold"])
        gp1, gp2 = st.columns(2)
        gp1.metric("Brecha de Precio Actual", f"${gap_actual:.2f}", help="Positivo = mas caro que la competencia")
        gp2.metric("Correlacion Precio-Unidades", f"{corr_precio_unidades:.3f}")
