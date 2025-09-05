# Streamlit – TGA/DTG Plotter with Noise Filtering & Event Detection
# Author: ChatGPT (GPT-5 Thinking)
# Description:
#   • Load one or multiple TGA datasets (CSV) with Temperature and Mass.
#   • Clean/normalize mass, smooth with Savitzky–Golay, compute derivative DTG (d(m%)/dT).
#   • Automatically detect thermal events (mass-loss steps) from DTG peaks.
#   • For each event, estimate Tonset/Tpeak/Tend and mass loss (%), plus final residue.
#   • Interactive Plotly charts (TGA and DTG), shaded event windows, and CSV exports.

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks

st.set_page_config(page_title="TGA • DTG • Eventos", page_icon="🔥", layout="wide")

# ------------------------------ Data Model ------------------------------- #
@dataclass
class Event:
    idx_left: int
    idx_peak: int
    idx_right: int
    Tonset: float
    Tpeak: float
    Tend: float
    mass_loss_pct: float

@dataclass
class SampleResult:
    name: str
    residue_pct: float
    events: List[Event]

# ------------------------------ Utilities -------------------------------- #
def robust_read_csv(file, decimal: str = '.') -> pd.DataFrame:
    """Read CSV trying default pandas parsing with specified decimal."""
    try:
        return pd.read_csv(file, decimal=decimal)
    except Exception:
        # fallback: try semicolon
        try:
            return pd.read_csv(file, sep=';', decimal=decimal)
        except Exception as e:
            raise e


def normalize_mass(mass: np.ndarray, assume_percent: bool, normalize_start: bool) -> np.ndarray:
    m = mass.astype(float)
    if not assume_percent:
        # assume raw mass in mg or arbitrary units: normalize to initial = 100%
        m = (m / max(m[0], 1e-12)) * 100.0
    elif normalize_start:
        m = (m / max(m[0], 1e-12)) * 100.0
    return m


def linear_drift_correction(T: np.ndarray, m_pct: np.ndarray, n_head: int = 50, n_tail: int = 50) -> np.ndarray:
    """Fit a line using first n_head and last n_tail points and subtract it from mass curve.
    Keeps mean level by re-adding average of endpoints. Useful for buoyancy drift.
    """
    n = len(T)
    idx = np.r_[np.arange(min(n_head, n)), np.arange(max(0, n - n_tail), n)]
    x = T[idx]
    y = m_pct[idx]
    if len(x) < 2:
        return m_pct
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    baseline = slope * T + intercept
    corrected = m_pct - (baseline - np.mean([y[0], y[-1]]))
    return corrected


def compute_dtg(T: np.ndarray, m_pct: np.ndarray, sg_window: int, sg_poly: int, deriv_smooth: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (m_smooth_pct, dtg) where dtg = d(m%)/dT (percentage points per °C)."""
    if deriv_smooth:
        # smooth and derive in one pass
        if sg_window % 2 == 0:
            sg_window += 1
        m_sm = savgol_filter(m_pct, window_length=max(5, sg_window), polyorder=sg_poly)
        dtg = savgol_filter(m_pct, window_length=max(5, sg_window), polyorder=sg_poly, deriv=1, delta=np.mean(np.diff(T)))
    else:
        m_sm = m_pct
        # finite differences
        dT = np.gradient(T)
        dtg = np.gradient(m_pct, dT)
    return m_sm, dtg


def detect_events(T: np.ndarray, dtg: np.ndarray, mass_pct: np.ndarray, prominence: float, width_pts: int, frac_height: float) -> List[Event]:
    """Detect negative DTG peaks as mass-loss events.
    - prominence: minimum prominence on -DTG
    - width_pts: minimum width in points
    - frac_height: fraction of peak magnitude to define left/right bounds
    """
    # Peaks of mass loss are negative in DTG; look for peaks on -dtg
    inv = -dtg
    pk, props = find_peaks(inv, prominence=prominence, width=width_pts)
    events: List[Event] = []
    for p in pk:
        amp = inv[p]
        thr = frac_height * amp
        # Left bound: move left until inv < thr or start reached
        i = int(p)
        while i > 0 and inv[i] > thr:
            i -= 1
        left = i
        # Right bound
        j = int(p)
        while j < len(inv) - 1 and inv[j] > thr:
            j += 1
        right = j
        if right <= left + 2:
            continue
        Ton, Tp, Tend = float(T[left]), float(T[p]), float(T[right])
        mass_loss = float(mass_pct[left] - mass_pct[right])
        if mass_loss < 0:
            # Skip pathological cases
            continue
        events.append(Event(left, int(p), right, Ton, Tp, Tend, mass_loss))
    return events

# ------------------------------- Sidebar --------------------------------- #
st.sidebar.header("Entrada de dados")
files = st.sidebar.file_uploader("Arquivos CSV (1 ou mais)", type=["csv", "txt"], accept_multiple_files=True)
if not files:
    st.info("Envie pelo menos 1 CSV contendo colunas de Temperatura e Massa.")
    st.stop()

st.sidebar.subheader("Mapeamento de colunas")
col_T = st.sidebar.text_input("Coluna de Temperatura", value="temperature")
col_m = st.sidebar.text_input("Coluna de Massa", value="mass")
assume_percent = st.sidebar.checkbox("Massa já em %", value=False, help="Desmarque se a massa estiver em mg ou unidades absolutas.")
normalize_start = st.sidebar.checkbox("Normalizar massa inicial para 100%", value=True)

st.sidebar.subheader("Pré-processamento & Derivada")
use_drift_corr = st.sidebar.checkbox("Correção linear de drift (pré/pós)", value=True)
n_head = st.sidebar.number_input("Pontos no início (baseline)", value=50, min_value=5, step=5)
n_tail = st.sidebar.number_input("Pontos no fim (baseline)", value=50, min_value=5, step=5)

use_savgol = st.sidebar.checkbox("Savitzky–Golay (suavizar/derivar)", value=True)
sg_window = st.sidebar.slider("Janela SG (pontos, ímpar)", 5, 201, 21, step=2)
sg_poly = st.sidebar.slider("Ordem do polinômio SG", 2, 5, 3)

st.sidebar.subheader("Detecção de eventos (DTG)")
prom = st.sidebar.number_input("Proeminência mínima (-DTG)", value=0.05, step=0.01, help="Em pontos percentuais por °C (pp/°C)")
width_pts = st.sidebar.number_input("Largura mínima (pontos)", value=10, step=1)
frac_height = st.sidebar.slider("Limite de janela (% do pico)", 1, 60, 10) / 100.0

st.sidebar.subheader("Outros")
T_min_global = st.sidebar.number_input("Temperatura mínima para plot", value=0.0, step=10.0)
T_max_global = st.sidebar.number_input("Temperatura máxima para plot", value=800.0, step=10.0)

# ------------------------------- Main UI --------------------------------- #
st.title("🔥 TGA com DTG, Filtro de Ruído e Eventos Térmicos")
st.caption("Sobrepõe curvas, calcula DTG, identifica eventos automaticamente e estima perdas de massa por etapa.")

fig_tga = go.Figure()
fig_dtg = go.Figure()

summary_rows: List[Dict] = []
processed_concat = []
results: List[SampleResult] = []

for f in files:
    name = f.name
    # Try both decimal separators
    try:
        raw = robust_read_csv(f, decimal='.')
    except Exception:
        f.seek(0)
        raw = robust_read_csv(f, decimal=',')

    if col_T not in raw.columns or col_m not in raw.columns:
        st.warning(f"{name}: não encontrei colunas '{col_T}' e '{col_m}'.")
        continue

    df = raw[[col_T, col_m]].dropna().astype(float)
    df.columns = ['T', 'm']
    df = df.sort_values('T')

    # Normalize mass to %
    m_pct = normalize_mass(df['m'].to_numpy(), assume_percent=assume_percent, normalize_start=normalize_start)

    # Drift correction (optional)
    if use_drift_corr:
        m_pct = linear_drift_correction(df['T'].to_numpy(), m_pct, int(n_head), int(n_tail))

    # Smooth + DTG
    m_sm, dtg = compute_dtg(df['T'].to_numpy(), m_pct, int(sg_window), int(sg_poly), deriv_smooth=use_savgol)

    # Event detection
    events = detect_events(df['T'].to_numpy(), dtg, m_sm, prominence=float(prom), width_pts=int(width_pts), frac_height=float(frac_height))

    # Residue at max temperature
    mask_window = (df['T'] >= T_min_global) & (df['T'] <= T_max_global)
    T_win = df['T'].to_numpy()[mask_window]
    m_win = m_sm[mask_window]
    residue = float(m_win[-1]) if len(m_win) > 0 else float(m_sm[-1])

    results.append(SampleResult(name=name, residue_pct=residue, events=events))

    # Plot TGA
    fig_tga.add_trace(go.Scatter(x=df['T'], y=m_sm, mode='lines', name=f"{name} – m%", line=dict(width=2)))

    # Shade event windows and annotate mass loss
    for k, ev in enumerate(events, start=1):
        fig_tga.add_vrect(x0=ev.Tonset, x1=ev.Tend, fillcolor='LightCoral', opacity=0.15, line_width=0,
                          annotation_text=f"E{k}: -{ev.mass_loss_pct:.1f}% @ {ev.Tpeak:.0f}°C", annotation_position='top left')

    # Plot DTG
    fig_dtg.add_trace(go.Scatter(x=df['T'], y=dtg, mode='lines', name=f"{name} – DTG", line=dict(width=2)))
    for ev in events:
        fig_dtg.add_vline(x=ev.Tpeak, line=dict(dash='dot', width=1))

    # Summary rows
    if events:
        cum_loss = sum([ev.mass_loss_pct for ev in events])
    else:
        cum_loss = 0.0
    summary_rows.append({
        'Amostra': name,
        'Eventos': len(events),
        'Perda total (%)': cum_loss,
        'Resíduo final (%)': residue,
    })

    # Processed export
    processed_concat.append(pd.DataFrame({
        'sample': name,
        'T_C': df['T'].to_numpy(),
        'mass_pct': m_sm,
        'DTG_pp_per_C': dtg,
    }))

# Layout & axes
fig_tga.update_layout(template='plotly_dark', height=520, xaxis_title='Temperatura (°C)', yaxis_title='Massa (%)', legend_title='Amostras')
fig_tga.update_xaxes(range=[T_min_global, T_max_global])

fig_dtg.update_layout(template='plotly_dark', height=420, xaxis_title='Temperatura (°C)', yaxis_title='DTG (pp/°C)', legend_title='Amostras')
fig_dtg.update_xaxes(range=[T_min_global, T_max_global])

st.plotly_chart(fig_tga, use_container_width=True)
st.plotly_chart(fig_dtg, use_container_width=True)

# ------------------------------- Tables & Export -------------------------- #
# Events table (long format)
rows = []
for r in results:
    for i, ev in enumerate(r.events, start=1):
        rows.append({
            'Amostra': r.name,
            'Evento': i,
            'Tonset (°C)': ev.Tonset,
            'Tpeak (°C)': ev.Tpeak,
            'Tend (°C)': ev.Tend,
            'Perda de massa (%)': ev.mass_loss_pct,
        })

if rows:
    st.subheader("Eventos detectados")
    df_events = pd.DataFrame(rows)
    st.dataframe(df_events, use_container_width=True)

if summary_rows:
    st.subheader("Resumo por amostra")
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True)

# Downloads
if processed_concat:
    comb = pd.concat(processed_concat, ignore_index=True)
    buf = io.StringIO(); comb.to_csv(buf, index=False)
    st.download_button("⬇️ Baixar curvas processadas (CSV)", buf.getvalue(), file_name="tga_processed_curves.csv", mime="text/csv")

if rows:
    evbuf = io.StringIO(); pd.DataFrame(rows).to_csv(evbuf, index=False)
    st.download_button("⬇️ Baixar eventos (CSV)", evbuf.getvalue(), file_name="tga_events.csv", mime="text/csv")

if summary_rows:
    sumbuf = io.StringIO(); pd.DataFrame(summary_rows).to_csv(sumbuf, index=False)
    st.download_button("⬇️ Baixar resumo (CSV)", sumbuf.getvalue(), file_name="tga_summary.csv", mime="text/csv")

# ------------------------------- Help Box -------------------------------- #
with st.expander("Notas & Boas Práticas"):
    st.markdown(
        """
        **Entrada**
        - CSV com colunas de `temperature` (°C) e `mass` (mg ou %). Mapeie os nomes nas opções.
        - Se a massa não estiver em %, o app normaliza a massa inicial para 100%.

        **Pré-processamento**
        - *Correção linear de drift*: ajusta uma linha usando os primeiros/últimos pontos e subtrai do sinal (útil para flutuações de empuxo).
        - *Savitzky–Golay*: define janela (ímpar) e ordem do polinômio; a derivada DTG pode ser calculada com smoothing.

        **Eventos (DTG)**
        - Os eventos são identificados como picos negativos em DTG; use `proeminência` e `largura` para controlar a detecção.
        - As janelas de evento são definidas onde `-DTG` cai abaixo de uma fração da altura do pico. A perda de massa é `m(Tonset) − m(Tend)`.

        **Saídas**
        - Tabelas com *Tonset, Tpeak, Tend, perda de massa* e *resíduo final*.
        - Exporte CSVs com curvas processadas, eventos e resumo.
        """
    )
