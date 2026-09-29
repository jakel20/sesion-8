"""
Módulo 7 — Series de Tiempo (Computación Avanzada · ET0197)
App interactiva de apoyo al cuaderno de Colab.

Ejecutar localmente:
    streamlit run app_series_tiempo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Series de Tiempo — Sensor IoT", page_icon="🌡️", layout="wide")

# -----------------------------------------------------------------
# Estilo visual (CSS + matplotlib)
# -----------------------------------------------------------------
COLORES = {
    "azul": "#2E6F9E",
    "naranja": "#E07A5F",
    "verde": "#3D9970",
    "morado": "#6C5B9E",
    "arena": "#F2CC8F",
    "gris": "#98A2B3",
    "oscuro": "#1F2937",
}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}

.hero {
    background: linear-gradient(120deg, #7A2E2E 0%, #E07A5F 45%, #2E6F9E 100%);
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 14px rgba(122, 46, 46, 0.22);
}
.hero .titulo {color: #FFFFFF; font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.hero .sub {color: rgba(255, 255, 255, 0.92); font-size: 0.98rem; margin-top: 0.4rem;}

.seccion {
    border-left: 5px solid #E07A5F;
    padding: 0.2rem 0 0.2rem 0.9rem;
    margin: 0.4rem 0 0.8rem 0;
}
.seccion .titulo {font-size: 1.35rem; font-weight: 700;}
.seccion .sub {font-size: 0.9rem; opacity: 0.75;}

div[data-testid="stMetric"] {
    background: rgba(224, 122, 95, 0.06);
    border: 1px solid rgba(224, 122, 95, 0.25);
    border-left: 5px solid #E07A5F;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
}
div[data-testid="stMetricLabel"] p {font-weight: 600;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(224, 122, 95, 0.10) 0%, rgba(46, 111, 158, 0.05) 100%);
}

button[data-baseweb="tab"] p {font-size: 0.95rem; font-weight: 600;}

.chip {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    margin: 0 0.35rem 0.4rem 0;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.chip-azul    {background: rgba(46, 111, 158, 0.14); color: #2E6F9E;}
.chip-naranja {background: rgba(224, 122, 95, 0.16); color: #C0583D;}
.chip-verde   {background: rgba(61, 153, 112, 0.16); color: #2E7D5B;}
.chip-morado  {background: rgba(108, 91, 158, 0.15); color: #5A4A8A;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#D0D7DE",
    "axes.grid": True,
    "grid.color": "#E6EAEE",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelcolor": "#344054",
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "legend.frameon": False,
    "font.size": 9,
})


def seccion(titulo, subtitulo=""):
    """Encabezado de cada pestaña."""
    st.markdown(
        f"<div class='seccion'><div class='titulo'>{titulo}</div>"
        f"<div class='sub'>{subtitulo}</div></div>",
        unsafe_allow_html=True,
    )


def mostrar_figura(fig):
    """Ajusta, muestra y libera la figura."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------------------------------------
# Generación de datos (cacheada por parámetros)
# -----------------------------------------------------------------
@st.cache_data
def generar_datos(n_dias, amplitud_estacional, nivel_ruido, pendiente_tendencia):
    n_horas = n_dias * 24
    tiempo = pd.date_range(start="2025-01-01", periods=n_horas, freq="h")
    t = np.arange(n_horas)

    tendencia = pendiente_tendencia * t
    estacionalidad = amplitud_estacional * np.sin(2 * np.pi * t / 24 - np.pi / 2)
    ruido = np.random.default_rng(42).normal(0, nivel_ruido, n_horas)

    temperatura = 22 + tendencia + estacionalidad + ruido
    serie = pd.Series(temperatura, index=tiempo, name="temperatura")
    return serie


def crear_ventanas(serie, tamano_ventana):
    valores = serie.values
    X, y = [], []
    for i in range(len(valores) - tamano_ventana):
        X.append(valores[i:i + tamano_ventana])
        y.append(valores[i + tamano_ventana])
    return np.array(X), np.array(y)


def mostrar_metricas(y_real, y_pred, extra=None):
    """Muestra MAE y RMSE como tarjetas (y una métrica adicional opcional)."""
    mae = mean_absolute_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred) ** 0.5
    columnas = st.columns(3 if extra else 2)
    columnas[0].metric("MAE", f"{mae:.3f}")
    columnas[1].metric("RMSE", f"{rmse:.3f}")
    if extra:
        columnas[2].metric(extra[0], extra[1])


# -----------------------------------------------------------------
# Sidebar — controles globales de la serie simulada
# -----------------------------------------------------------------
st.sidebar.title("⚙️ Sensor simulado")
st.sidebar.caption("DHT22 · temperatura horaria")

st.sidebar.markdown("##### 📅 Duración")
n_dias = st.sidebar.slider("Días simulados", min_value=10, max_value=90, value=45, step=5)

st.sidebar.markdown("##### 🧩 Componentes de la serie")
amplitud = st.sidebar.slider("Amplitud del ciclo diario (°C)", min_value=0.0, max_value=10.0, value=4.0, step=0.5)
ruido = st.sidebar.slider("Nivel de ruido", min_value=0.0, max_value=3.0, value=0.6, step=0.1)
tendencia_pendiente = st.sidebar.slider("Pendiente de la tendencia", min_value=-0.05, max_value=0.05, value=0.01, step=0.005)

serie = generar_datos(n_dias, amplitud, ruido, tendencia_pendiente)

st.sidebar.divider()
st.sidebar.caption(
    "Estos controles simulan un sensor DHT22 (temperatura). "
    "Cámbialos y observa cómo se transforma cada gráfica en las pestañas."
)

# -----------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------
st.markdown(
    "<div class='hero'><div class='titulo'>🌡️ Series de Tiempo — Sensor IoT interactivo</div>"
    "<div class='sub'>Computación Avanzada · ET0197 — Acompaña al cuaderno "
    "<i>Módulo 7 — Series de Tiempo</i></div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span class='chip chip-naranja'>📈 Tendencia</span>"
    "<span class='chip chip-verde'>🔁 Estacionalidad</span>"
    "<span class='chip chip-morado'>🎲 Ruido</span>"
    "<span class='chip chip-azul'>🔮 Pronóstico</span>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Componentes",
    "2️⃣ ACF / PACF",
    "3️⃣ Ventanas deslizantes",
    "4️⃣ Modelos clásicos",
])

# -----------------------------------------------------------------
# Tab 1: Componentes de la serie
# -----------------------------------------------------------------
with tab1:
    seccion("Tendencia, estacionalidad y ruido", "Las piezas que, sumadas, forman la serie observada.")
    st.write(
        "Ajusta los controles de la barra lateral y observa cómo cada componente "
        "afecta la forma de la serie y su descomposición."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Horas simuladas", len(serie))
    m2.metric("Media (°C)", f"{serie.mean():.2f}")
    m3.metric("Mínimo (°C)", f"{serie.min():.2f}")
    m4.metric("Máximo (°C)", f"{serie.max():.2f}")

    fig1, ax1 = plt.subplots(figsize=(11, 3.2))
    ax1.plot(serie.index, serie.values, color=COLORES["naranja"], linewidth=0.9, alpha=0.85, label="Temperatura")
    ax1.plot(serie.index, serie.rolling(24).mean().values, color=COLORES["azul"], linewidth=2,
             label="Media móvil 24 h (referencia)")
    ax1.set_title("Serie simulada — Temperatura (°C)")
    ax1.set_ylabel("°C")
    ax1.legend(fontsize=8, loc="upper left")
    mostrar_figura(fig1)

    descomposicion = seasonal_decompose(serie, model="additive", period=24)
    componentes = [
        ("Observada", descomposicion.observed, COLORES["naranja"]),
        ("Tendencia", descomposicion.trend, COLORES["azul"]),
        ("Estacionalidad (período 24 h)", descomposicion.seasonal, COLORES["verde"]),
        ("Residuo", descomposicion.resid, COLORES["morado"]),
    ]
    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for ax, (nombre, comp, color) in zip(axes2, componentes):
        if nombre == "Residuo":
            ax.scatter(comp.index, comp.values, s=4, color=color, alpha=0.6)
            ax.axhline(0, color=COLORES["oscuro"], linewidth=0.8)
        else:
            ax.plot(comp.index, comp.values, color=color, linewidth=1.3)
        ax.set_title(nombre, loc="left", fontsize=10)
    fig2.suptitle("Descomposición aditiva", fontweight="bold", fontsize=12)
    mostrar_figura(fig2)

    st.info(
        "💡 Sube el **ruido** al máximo: el residuo debería crecer y volverse menos estructurado. "
        "Sube la **amplitud del ciclo diario**: la estacionalidad se hace más pronunciada."
    )

# -----------------------------------------------------------------
# Tab 2: ACF / PACF
# -----------------------------------------------------------------
with tab2:
    seccion("Autocorrelación (ACF) y autocorrelación parcial (PACF)",
            "¿Cuánto se parece la serie a sí misma desplazada k horas?")
    n_lags = st.slider("Número de lags a mostrar", min_value=12, max_value=168, value=72, step=12)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            plot_acf(serie, lags=n_lags, ax=ax3, color=COLORES["azul"],
                     vlines_kwargs={"colors": COLORES["azul"]})
            for k in range(24, n_lags + 1, 24):
                ax3.axvline(k, color=COLORES["naranja"], linestyle=":", linewidth=1.2)
            ax3.set_title("ACF  (líneas punteadas: múltiplos de 24 h)")
            ax3.set_xlabel("Lag (horas)")
            mostrar_figura(fig3)
    with col2:
        with st.container(border=True):
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            plot_pacf(serie, lags=n_lags, ax=ax4, method="ywm", color=COLORES["verde"],
                      vlines_kwargs={"colors": COLORES["verde"]})
            ax4.set_title("PACF")
            ax4.set_xlabel("Lag (horas)")
            mostrar_figura(fig4)

    st.info(
        "💡 Con el ciclo diario activo deberías ver picos de ACF cada 24 horas. "
        "Baja la amplitud del ciclo diario a 0 en la barra lateral y observa cómo desaparecen los picos."
    )

# -----------------------------------------------------------------
# Tab 3: Ventanas deslizantes
# -----------------------------------------------------------------
with tab3:
    seccion("De serie de tiempo a datos tabulares (X, y)",
            "Las últimas horas se convierten en predictores del siguiente valor.")
    tamano_ventana = st.slider("Tamaño de la ventana (horas pasadas usadas como predictor)",
                               min_value=3, max_value=72, value=24, step=3)

    X, y = crear_ventanas(serie, tamano_ventana)

    m1, m2 = st.columns(2)
    m1.metric("Muestras generadas", X.shape[0])
    m2.metric("Predictores por muestra", X.shape[1])

    # Ilustración de una ventana
    n_vista = min(len(serie), tamano_ventana + 24)
    valores_vista = serie.values[:n_vista]
    fig_v, ax_v = plt.subplots(figsize=(11, 2.8))
    ax_v.plot(np.arange(n_vista), valores_vista, color=COLORES["gris"], linewidth=1.2, marker="o", markersize=3)
    ax_v.axvspan(-0.5, tamano_ventana - 0.5, color=COLORES["azul"], alpha=0.12, label="Ventana → X (predictores)")
    ax_v.scatter([tamano_ventana], [valores_vista[tamano_ventana]], color=COLORES["naranja"], s=90,
                 zorder=5, edgecolors="white", linewidths=1.5, label="Siguiente valor → y")
    ax_v.set_xlabel("Hora")
    ax_v.set_ylabel("°C")
    ax_v.set_title("Primera muestra: así se construye cada fila de la tabla")
    ax_v.legend(fontsize=8, loc="upper right")
    mostrar_figura(fig_v)

    with st.container(border=True):
        st.markdown("**Primeras 5 filas de la tabla (X, y)**")
        ejemplo = pd.DataFrame(X[:5], columns=[f"t-{tamano_ventana - i}" for i in range(tamano_ventana)])
        ejemplo["y (siguiente valor)"] = y[:5]
        st.dataframe(ejemplo.round(2), use_container_width=True)

    corte = int(len(X) * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    modelo = LinearRegression().fit(X_train, y_train)
    pred = modelo.predict(X_test)

    fig5, ax5 = plt.subplots(figsize=(11, 3.2))
    ax5.plot(y_test, label="Real", color=COLORES["oscuro"], linewidth=1.6)
    ax5.plot(pred, label="Predicho (regresión sobre ventanas)", linestyle="--",
             color=COLORES["naranja"], linewidth=1.6)
    ax5.fill_between(np.arange(len(y_test)), y_test, pred, color=COLORES["naranja"], alpha=0.12)
    ax5.legend(fontsize=8, loc="upper left")
    ax5.set_xlabel("Hora dentro del conjunto de prueba")
    ax5.set_ylabel("°C")
    ax5.set_title("Regresión lineal entrenada sobre ventanas deslizantes")
    mostrar_figura(fig5)

    mostrar_metricas(y_test, pred, extra=("Muestras de prueba", len(y_test)))
    st.info(
        "💡 Prueba ventanas muy pequeñas (3-6 horas) vs. muy grandes (48-72 horas) "
        "y observa cómo cambia el error."
    )

# -----------------------------------------------------------------
# Tab 4: Modelos clásicos de pronóstico
# -----------------------------------------------------------------
with tab4:
    seccion("Media móvil, suavizado exponencial y ARIMA",
            "Modelos clásicos de pronóstico evaluados sobre las últimas horas de la serie.")

    col_cfg1, col_cfg2 = st.columns(2)
    horas_test = col_cfg1.slider("Horas a pronosticar (conjunto de prueba)", min_value=24, max_value=240, value=120, step=24)
    train = serie.iloc[:-horas_test]
    test = serie.iloc[-horas_test:]

    modelo_elegido = col_cfg2.selectbox(
        "Modelo",
        ["Media móvil", "Suavizado exponencial simple", "Holt-Winters", "ARIMA", "SARIMA"],
    )

    with st.container(border=True):
        if modelo_elegido == "Media móvil":
            ventana_mm = st.slider("Tamaño de la ventana de la media móvil", 6, 72, 24, step=6)
            historial = list(train.values[-ventana_mm:])
            pred_vals = []
            for _ in range(horas_test):
                siguiente = np.mean(historial[-ventana_mm:])
                pred_vals.append(siguiente)
                historial.append(siguiente)
            pronostico = pd.Series(pred_vals, index=test.index)

        elif modelo_elegido == "Suavizado exponencial simple":
            alpha = st.slider("Alpha (peso de la observación más reciente)", 0.05, 0.95, 0.3, step=0.05)
            modelo_fit = SimpleExpSmoothing(train, initialization_method="estimated").fit(smoothing_level=alpha)
            pronostico = modelo_fit.forecast(horas_test)

        elif modelo_elegido == "Holt-Winters":
            st.caption("Holt-Winters aditivo · tendencia + estacionalidad de 24 h (parámetros estimados automáticamente).")
            modelo_fit = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=24,
                initialization_method="estimated",
            ).fit()
            pronostico = modelo_fit.forecast(horas_test)

        elif modelo_elegido == "ARIMA":
            col_p, col_d, col_q = st.columns(3)
            with col_p:
                p = st.slider("p (orden autorregresivo)", 0, 4, 4)
            with col_d:
                d = st.slider("d (diferenciación)", 0, 2, 1)
            with col_q:
                q = st.slider("q (orden de media móvil)", 0, 4, 2)
            with st.spinner("Ajustando ARIMA..."):
                modelo_fit = ARIMA(train, order=(p, d, q)).fit()
            pronostico = modelo_fit.forecast(horas_test)
            st.caption(
                "💡 Este modelo NO tiene noción explícita de estacionalidad: solo puede "
                "'imitar' el ciclo de 24 horas subiendo p. Compáralo con SARIMA más abajo."
            )

        else:  # SARIMA
            st.markdown("**Parte no estacional** — igual que ARIMA, sobre los rezagos cercanos:")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                p = st.slider("p", 0, 3, 2)
            with col_b:
                d = st.slider("d", 0, 2, 0)
            with col_c:
                q = st.slider("q", 0, 3, 2)

            st.markdown("**Parte estacional** — los mismos conceptos, aplicados cada `s` horas:")
            col_d, col_e, col_f, col_g = st.columns(4)
            with col_d:
                P = st.slider("P (AR estacional)", 0, 2, 1)
            with col_e:
                D = st.slider("D (diferenciación estacional)", 0, 1, 0)
            with col_f:
                Q = st.slider("Q (MA estacional)", 0, 2, 1)
            with col_g:
                s = st.slider("s (período estacional)", 6, 48, 24, step=6)

            with st.spinner("Ajustando SARIMA..."):
                modelo_fit = SARIMAX(
                    train, order=(p, d, q), seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            pronostico = modelo_fit.forecast(horas_test)

            if s != 24:
                st.warning(
                    f"⚠️ Declaraste s={s}, pero el ciclo real del sensor es de 24 horas. "
                    "Observa cómo empeora el error cuando el período estacional no coincide con el real."
                )
            else:
                st.caption(
                    "💡 s=24 le dice al modelo 'compara cada hora con la misma hora del día anterior'. "
                    "Prueba subir P o Q, o cambiar s, y observa el efecto en el error."
                )

    contexto = train.iloc[-min(len(train), 72):]
    fig6, ax6 = plt.subplots(figsize=(11, 4))
    ax6.plot(contexto.index, contexto.values, color=COLORES["gris"], linewidth=1.3,
             label="Entrenamiento (últimas 72 h)")
    ax6.axvspan(test.index[0], test.index[-1], color=COLORES["arena"], alpha=0.18, label="Periodo de prueba")
    ax6.plot(test.index, test.values, label="Real", color=COLORES["oscuro"], linewidth=1.8)
    ax6.plot(test.index, pronostico.values, label=f"Pronóstico ({modelo_elegido})",
             linestyle="--", color=COLORES["naranja"], linewidth=2)
    ax6.fill_between(test.index, test.values, pronostico.values, color=COLORES["naranja"], alpha=0.12)
    ax6.set_ylabel("°C")
    ax6.legend(fontsize=8, loc="upper left", ncol=2)
    ax6.set_title(f"Pronóstico — {modelo_elegido}")
    mostrar_figura(fig6)

    mostrar_metricas(test.values, pronostico.values, extra=("Horas pronosticadas", horas_test))
    st.info(
        "💡 Compara los cinco modelos con los mismos parámetros del sensor. "
        "¿Cuál mantiene el error más bajo cuando subes el ruido en la barra lateral? "
        "¿SARIMA logra superar a Holt-Winters?"
    )
