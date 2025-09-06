# streamlit_tga_plotter.py
# -*- coding: utf-8 -*-
"""
App Streamlit para PLOTAR dados de TGA/DTG a partir de TXT/CSV.
Recursos principais:
- Upload de 1 ou vários arquivos
- Detecção automática de separador (espaço, TAB, ';', ',') e vírgula decimal
- Mapeamento automático (e manual, se necessário) de colunas: Temperature / Mass (ou Weight)
- Normalização opcional da massa para 100% no início; cálculo de DTG (= d(m%)/dT)
- Smoothing: média móvel embutida; Savitzky–Golay (se SciPy estiver disponível)
- Correção linear de drift (opcional) com base em pontos iniciais e finais
- Plotagem (matplotlib) de TGA (m% vs T) e DTG (derivada)
- Exportação dos dados processados por arquivo

Execução local:
    streamlit run streamlit_tga_plotter.py
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Tente importar Savitzky-Golay e find_peaks; se não houver SciPy, seguimos com alternativas
try:
    from scipy.signal import savgol_filter, find_peaks
    SCIPY_OK = True
except Exception:
    savgol_filter = None
    find_peaks = None
    SCIPY_OK = False

import matplotlib.pyplot as plt


# --------------------- Leitura e Padronização ---------------------

KNOWN_HEADER_TOKENS = {
    "time", "t", "tempo",
    "temp", "temperature", "temperatura",
    "weight", "weight%", "mass", "massa", "mass%", "mass_pct"
}
SEPARATORS = [r"\s+", "\t", ";", ","]


def has_decimal_comma(text: str) -> bool:
    return bool(re.search(r"\d+,\d+", text))


def find_header_row(text: str) -> int:
    lines = text.splitlines()
    max_check = min(len(lines), 200)
    for i in range(max_check):
        tokens = re.split(r"[;\t, ]+", lines[i].strip())
        tokens_lc = {t.lower() for t in tokens if t}
        if tokens_lc & KNOWN_HEADER_TOKENS and len(tokens_lc) >= 2:
            return i
    return 0


def try_read_with_sep(text: str, header_row: int, sep: str, decimal: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            engine="python",
            header=0,
            skiprows=header_row,
            decimal=decimal
        )
    except Exception:
        return None


def dedupe_columns(cols: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in cols:
        c = str(c).strip()
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(f"{c}_{counts[c]}" if counts[c] > 1 else c)
    return new_cols


def robust_read_to_df(file_bytes: bytes, decimal_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str], int, str]:
    """
    Retorna (df_lido, mapping_inicial, header_row, sep_usado).
    mapping_inicial tenta mapear ['temperature', 'mass', 'mass_pct', 'time'].
    """
    text = file_bytes.decode("utf-8", errors="replace")
    header_row = find_header_row(text)
    decimal = decimal_hint or ("," if has_decimal_comma(text) else ".")

    # tente vários separadores
    for sep in SEPARATORS:
        df = try_read_with_sep(text, header_row, sep, decimal)
        if df is None or df.empty:
            continue
        df.columns = dedupe_columns(list(df.columns))
        mapping = auto_map_columns(df)
        return df, mapping, header_row, sep

    # fallback: largura fixa
    try:
        df = pd.read_fwf(io.StringIO("\n".join(text.splitlines()[header_row:])), header=0)
        df.columns = dedupe_columns(list(df.columns))
        mapping = auto_map_columns(df)
        return df, mapping, header_row, "fwf"
    except Exception as e:
        raise ValueError(f"Falha na leitura: {e}")


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Retorna um dicionário {padrao: colunaEncontrada}, onde padrao in ["temperature","mass","mass_pct","time"].
    Aceita variações como Weight/Mass/Temperatura/Time etc. Lida com duplicatas de Weight.
    """
    out: Dict[str, str] = {}
    # candidatos por nome
    for c in df.columns:
        lc = c.lower().strip()
        if lc.startswith("temp"):
            out.setdefault("temperature", c)
        elif lc.startswith("time") or lc in {"t", "tempo"}:
            out.setdefault("time", c)
        elif lc.startswith("mass") or lc.startswith("massa") or lc.startswith("weight"):
            # decidir mais tarde
            pass

    # identificar colunas de massa/peso
    weight_like = [c for c in df.columns if c not in out.values()
                   and (c.lower().startswith("mass") or c.lower().startswith("massa") or c.lower().startswith("weight"))]

    # Heurística: coluna com valores ~100 é porcentagem; outra é massa absoluta
    perc_col = None
    gram_col = None
    for c in weight_like:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if v.empty:
            continue
        if (v.max() > 50) or (len(v) > 0 and v.iloc[0] >= 50):
            perc_col = perc_col or c
        else:
            gram_col = gram_col or c

    # fallbacks
    if gram_col is None and weight_like:
        gram_col = weight_like[0]
        if len(weight_like) > 1:
            perc_col = perc_col or weight_like[1]

    if gram_col:
        out["mass"] = gram_col
    if perc_col:
        out["mass_pct"] = perc_col
    return out


# --------------------- Processamento ---------------------

@dataclass
class TGAOptions:
    assume_percent: bool = False
    normalize_start: bool = True
    drift_correction: bool = False
    drift_head_pts: int = 50
    drift_tail_pts: int = 50
    smoothing: str = "Nenhum"  # Nenhum | MediaMovel | Savitzky-Golay
    ma_window: int = 11
    sg_window: int = 21
    sg_poly: int = 3


def to_percent_mass(mass: np.ndarray, assume_percent: bool, normalize_start: bool) -> np.ndarray:
    m = mass.astype(float)
    if not assume_percent:
        m = (m / max(m[0], 1e-12)) * 100.0
    elif normalize_start:
        m = (m / max(m[0], 1e-12)) * 100.0
    return m


def linear_drift_correction(temp: np.ndarray, y: np.ndarray, n_head: int, n_tail: int) -> np.ndarray:
    n = len(y)
    n_head = max(2, min(n_head, n//2))
    n_tail = max(2, min(n_tail, n//2))
    idx = np.r_[np.arange(n_head), np.arange(n - n_tail, n)]
    coef = np.polyfit(temp[idx], y[idx], 1)
    baseline = np.polyval(coef, temp)
    return y - baseline + y[0]  # preserva o valor inicial


def smooth_signal(y: np.ndarray, opts: TGAOptions, deriv: bool = False, x: Optional[np.ndarray] = None) -> np.ndarray:
    # y: já em m%
    if opts.smoothing == "Nenhum":
        if deriv:
            # derivada numérica d(y)/dT
            if x is None:
                x = np.arange(len(y))
            return np.gradient(y, x)
        return y
    elif opts.smoothing == "MediaMovel":
        w = max(3, int(opts.ma_window) | 1)  # ímpar
        pad = w // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        c = np.ones(w) / w
        ys = np.convolve(ypad, c, mode="valid")
        if deriv:
            if x is None:
                x = np.arange(len(ys))
            return np.gradient(ys, x)
        return ys
    elif opts.smoothing == "Savitzky-Golay":
        if not SCIPY_OK:
            # fallback para média móvel
            return smooth_signal(y, TGAOptions(**{**opts.__dict__, "smoothing": "MediaMovel"}), deriv, x)
        w = max(5, int(opts.sg_window) | 1)
        p = max(2, int(opts.sg_poly))
        if deriv:
            if x is None:
                x = np.arange(len(y))
            # Savitzky-Golay com derivada primeira em relação a x (Temperatura)
            # Precisamos do passo médio de x para escalar corretamente
            dx = np.mean(np.diff(x))
            if dx <= 0:
                dx = 1.0
            return savgol_filter(y, window_length=w, polyorder=p, deriv=1, delta=dx, mode="interp")
        else:
            return savgol_filter(y, window_length=w, polyorder=p, mode="interp")
    return y


def process_single(df: pd.DataFrame, mapping: Dict[str, str], opts: TGAOptions) -> pd.DataFrame:
    # Garantir que temos temperatura e massa (ou mass_pct)
    if "temperature" not in mapping:
        raise ValueError("Coluna de temperatura não encontrada. Ajuste o mapeamento.")
    if "mass" not in mapping and "mass_pct" not in mapping:
        raise ValueError("Coluna de massa/porcentagem não encontrada. Ajuste o mapeamento.")

    T = pd.to_numeric(df[mapping["temperature"]], errors="coerce").to_numpy()
    if "mass" in mapping:
        m_raw = pd.to_numeric(df[mapping["mass"]], errors="coerce").to_numpy()
        m_pct = to_percent_mass(m_raw, opts.assume_percent, opts.normalize_start)
    else:
        m_pct = pd.to_numeric(df[mapping["mass_pct"]], errors="coerce").to_numpy()
        # Se o usuário marcou "normalize_start", reescale para 100% no início
        if opts.normalize_start:
            m_pct = (m_pct / max(m_pct[0], 1e-12)) * 100.0

    # Correção de drift
    if opts.drift_correction:
        m_pct = linear_drift_correction(T, m_pct, opts.drift_head_pts, opts.drift_tail_pts)

    # Smoothing e DTG
    m_pct_s = smooth_signal(m_pct, opts, deriv=False, x=T)
    dtg = -smooth_signal(m_pct_s, opts, deriv=True, x=T)  # sinal negativo para perdas de massa

    out = pd.DataFrame({
        "Temperature": T,
        "Mass_pct": m_pct_s,
        "DTG_(-%/°C)": dtg
    })
    return out.dropna()


# --------------------- UI ---------------------

st.set_page_config(page_title="TGA/DTG Plotter", layout="wide")
st.title("📈 TGA/DTG Plotter (TXT/CSV)")

st.markdown(
    """
Carregue arquivos de TGA (TXT/CSV) contendo **Temperatura** e **Massa** (ou **Massa %**).
O app detecta colunas automaticamente e permite ajustes manuais.
"""
)

with st.sidebar:
    st.header("Opções de Importação")
    dec_choice = st.selectbox("Separador decimal", ["Auto", ".", ","], index=0)
    decimal_hint = None if dec_choice == "Auto" else dec_choice

    st.header("Pré-processamento")
    assume_percent = st.checkbox("Valores de massa já estão em %", value=False)
    normalize_start = st.checkbox("Normalizar massa para 100% no início", value=True)
    drift_correction = st.checkbox("Correção linear de drift", value=False)
    drift_head_pts = st.number_input("Pontos iniciais (drift)", 10, 1000, 50, 1)
    drift_tail_pts = st.number_input("Pontos finais (drift)", 10, 1000, 50, 1)

    st.header("Suavização (smoothing)")
    smoothing = st.selectbox("Método", ["Nenhum", "MediaMovel", "Savitzky-Golay" if SCIPY_OK else "Savitzky-Golay (SciPy ausente)"])
    if smoothing == "MediaMovel":
        ma_window = st.number_input("Janela média móvel (ímpar)", 3, 501, 11, 2)
        sg_window, sg_poly = 21, 3
    else:
        ma_window = 11
        if SCIPY_OK and smoothing.startswith("Savitzky-Golay"):
            sg_window = st.number_input("Janela SG (ímpar)", 5, 501, 21, 2)
            sg_poly = st.number_input("Ordem do polinômio SG", 2, 7, 3, 1)
        else:
            sg_window, sg_poly = 21, 3

opts = TGAOptions(
    assume_percent=assume_percent,
    normalize_start=normalize_start,
    drift_correction=drift_correction,
    drift_head_pts=int(drift_head_pts),
    drift_tail_pts=int(drift_tail_pts),
    smoothing="Savitzky-Golay" if (SCIPY_OK and smoothing.startswith("Savitzky-Golay")) else ("MediaMovel" if smoothing=="MediaMovel" else "Nenhum"),
    ma_window=int(ma_window),
    sg_window=int(sg_window),
    sg_poly=int(sg_poly),
)

uploaded_files = st.file_uploader("Envie 1 ou mais arquivos .txt/.csv", type=["txt","csv"], accept_multiple_files=True)

if uploaded_files:
    all_processed: Dict[str, pd.DataFrame] = {}
    mapping_per_file: Dict[str, Dict[str,str]] = {}

    # Primeiro: ler e mapear
    st.subheader("Mapeamento de Colunas")
    for f in uploaded_files:
        df_raw, mapping_guess, header_row, sep_used = robust_read_to_df(f.getvalue(), decimal_hint=decimal_hint)
        st.markdown(f"**{f.name}** — cabeçalho na linha {header_row+1} • sep: `{sep_used}`")

        # Interface de mapeamento manual se necessário
        cols = list(df_raw.columns)
        col_temp = st.selectbox(f"Coluna de **Temperatura** ({f.name})", cols, index=cols.index(mapping_guess["temperature"]) if "temperature" in mapping_guess else 0, key=f"{f.name}_temp")
        col_mass = st.selectbox(f"Coluna de **Massa (g ou mg)** ({f.name})", ["(nenhuma)"]+cols, index=(cols.index(mapping_guess["mass"])+1) if "mass" in mapping_guess else 0, key=f"{f.name}_mass")
        col_mpct = st.selectbox(f"Coluna de **Massa %** ({f.name})", ["(nenhuma)"]+cols, index=(cols.index(mapping_guess["mass_pct"])+1) if "mass_pct" in mapping_guess else 0, key=f"{f.name}_mpct")

        mapping = {"temperature": col_temp}
        if col_mass != "(nenhuma)": mapping["mass"] = col_mass
        if col_mpct != "(nenhuma)": mapping["mass_pct"] = col_mpct
        mapping_per_file[f.name] = mapping

        # Processa
        try:
            df_proc = process_single(df_raw, mapping, opts)
            all_processed[f.name] = df_proc
        except Exception as e:
            st.error(f"{f.name}: erro no processamento — {e}")

    # Gráficos
    if all_processed:
        st.subheader("Gráficos")

        # TGA
        fig1, ax1 = plt.subplots()
        for name, d in all_processed.items():
            ax1.plot(d["Temperature"], d["Mass_pct"], label=name)
        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Massa (%)")
        ax1.set_title("TGA — Massa (%) vs Temperatura")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()
        st.pyplot(fig1, clear_figure=True)

        # DTG
        fig2, ax2 = plt.subplots()
        for name, d in all_processed.items():
            ax2.plot(d["Temperature"], d["DTG_(-%/°C)"], label=name)
        ax2.set_xlabel("Temperatura (°C)")
        ax2.set_ylabel("-d(M%)/dT (%/°C)")
        ax2.set_title("DTG — Derivada da Massa (%)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

        # Exibição e Downloads
        st.subheader("Dados Processados e Exportação")
        for name, d in all_processed.items():
            st.markdown(f"**{name}** — {len(d)} pontos")
            st.dataframe(d.head(30), use_container_width=True)
            buf = io.StringIO()
            d.to_csv(buf, index=False)
            st.download_button(
                label=f"⬇️ CSV processado — {name}",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"{name.rsplit('.',1)[0]}_processado.csv",
                mime="text/csv",
            )
else:
    st.info("Envie um ou mais arquivos para visualizar TGA/DTG.")



