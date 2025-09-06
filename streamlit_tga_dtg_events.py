# streamlit_tga_plotter.py
# -*- coding: utf-8 -*-
"""
App Streamlit para PLOTAR dados de TGA/DTG a partir de TXT/CSV.
Novidades:
- Cortar início por temperatura (Tmin) e/ou por N pontos
- Ajuste do início em Y (re-escalar para Y0 alvo e offset opcional)
- Estilo por série: cor, espessura e rótulo de legenda; incluir/excluir
- Ajustes globais: tamanhos de fonte (título, eixos, ticks, legenda) e limites de eixos
- Exportação em alta resolução (PNG/SVG) com DPI configurável
- Gráficos combinados (TGA e DTG) com várias séries no mesmo plot

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

# Tente importar Savitzky-Golay; se não houver SciPy, usamos média móvel
try:
    from scipy.signal import savgol_filter
    SCIPY_OK = True
except Exception:
    savgol_filter = None
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
    text = file_bytes.decode("utf-8", errors="replace")
    header_row = find_header_row(text)
    decimal = decimal_hint or ("," if has_decimal_comma(text) else ".")

    for sep in SEPARATORS:
        df = try_read_with_sep(text, header_row, sep, decimal)
        if df is None or df.empty:
            continue
        df.columns = dedupe_columns(list(df.columns))
        mapping = auto_map_columns(df)
        return df, mapping, header_row, sep

    # fallback: largura fixa
    df = pd.read_fwf(io.StringIO("\n".join(text.splitlines()[header_row:])), header=0)
    df.columns = dedupe_columns(list(df.columns))
    mapping = auto_map_columns(df)
    return df, mapping, header_row, "fwf"


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc.startswith("temp"):
            out.setdefault("temperature", c)
        elif lc.startswith("time") or lc in {"t", "tempo"}:
            out.setdefault("time", c)
        elif lc.startswith("mass") or lc.startswith("massa") or lc.startswith("weight"):
            pass

    weight_like = [c for c in df.columns if c not in out.values()
                   and (c.lower().startswith("mass") or c.lower().startswith("massa") or c.lower().startswith("weight"))]

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
    smoothing: str = "Nenhum"  # Nenhum | MediaMovel | Savitzky-Golay
    ma_window: int = 11
    sg_window: int = 21
    sg_poly: int = 3

@dataclass
class TrimAlignOptions:
    cut_Tmin: Optional[float] = None
    cut_Nfirst: int = 0
    y_target_start: Optional[float] = 100.0  # re-escala para que o 1º ponto = alvo (None desativa)
    y_offset: float = 0.0  # offset aditivo em m% (não afeta DTG)


def to_percent_mass(mass: np.ndarray, assume_percent: bool, normalize_start: bool) -> np.ndarray:
    m = mass.astype(float)
    if not assume_percent:
        m = (m / max(m[0], 1e-12)) * 100.0
    elif normalize_start:
        m = (m / max(m[0], 1e-12)) * 100.0
    return m


def smooth_signal(y: np.ndarray, opts: TGAOptions, deriv: bool = False, x: Optional[np.ndarray] = None) -> np.ndarray:
    if opts.smoothing == "Nenhum":
        if deriv:
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
            dx = np.mean(np.diff(x))
            if dx <= 0:
                dx = 1.0
            return savgol_filter(y, window_length=w, polyorder=p, deriv=1, delta=dx, mode="interp")
        else:
            return savgol_filter(y, window_length=w, polyorder=p, mode="interp")
    return y


def process_single(df: pd.DataFrame, mapping: Dict[str, str], tga_opts: TGAOptions, trim_opts: TrimAlignOptions) -> pd.DataFrame:
    if "temperature" not in mapping:
        raise ValueError("Coluna de temperatura não encontrada. Ajuste o mapeamento.")
    if "mass" not in mapping and "mass_pct" not in mapping:
        raise ValueError("Coluna de massa/porcentagem não encontrada. Ajuste o mapeamento.")

    T = pd.to_numeric(df[mapping["temperature"]], errors="coerce").to_numpy()

    if "mass" in mapping:
        m_raw = pd.to_numeric(df[mapping["mass"]], errors="coerce").to_numpy()
        m_pct = to_percent_mass(m_raw, tga_opts.assume_percent, tga_opts.normalize_start)
    else:
        m_pct = pd.to_numeric(df[mapping["mass_pct"]], errors="coerce").to_numpy()
        if tga_opts.normalize_start:
            m_pct = (m_pct / max(m_pct[0], 1e-12)) * 100.0

    # --- Corte inicial ---
    mask = np.ones_like(T, dtype=bool)
    if trim_opts.cut_Tmin is not None:
        mask &= (T >= float(trim_opts.cut_Tmin))
    if trim_opts.cut_Nfirst > 0:
        idxs = np.where(mask)[0]
        if len(idxs) > trim_opts.cut_Nfirst:
            # zera os primeiros N pontos válidos
            mask[idxs[:trim_opts.cut_Nfirst]] = False
    T = T[mask]
    m_pct = m_pct[mask]

    # --- Suavização e DTG (antes dos ajustes Y para preservar forma básica) ---
    m_pct_s = smooth_signal(m_pct, tga_opts, deriv=False, x=T)
    dtg = -smooth_signal(m_pct_s, tga_opts, deriv=True, x=T)

    # --- Ajuste do início em Y ---
    if trim_opts.y_target_start is not None and len(m_pct_s) > 0:
        current = m_pct_s[0]
        if abs(current) > 1e-12:
            scale = (float(trim_opts.y_target_start) / current)
            m_pct_s = m_pct_s * scale
            dtg = dtg * scale  # derivada escala junto
    if abs(trim_opts.y_offset) > 0:
        m_pct_s = m_pct_s + float(trim_opts.y_offset)
        # Offset não altera derivada

    out = pd.DataFrame({
        "Temperature": T,
        "Mass_pct": m_pct_s,
        "DTG_(-%/°C)": dtg
    })
    return out.dropna()


# --------------------- UI ---------------------

st.set_page_config(page_title="TGA/DTG Plotter", layout="wide")
st.title("📈 TGA/DTG Plotter (TXT/CSV) — v2")

st.markdown(
    """
Envie arquivos de TGA (TXT/CSV) com **Temperatura** e **Massa** (ou **Massa %**).
Mapeie colunas, corte o início em X/Y, ajuste estilo e exporte em alta resolução.
"""
)

with st.sidebar:
    st.header("Importação")
    dec_choice = st.selectbox("Separador decimal", ["Auto", ".", ","], index=0)
    decimal_hint = None if dec_choice == "Auto" else dec_choice

    st.header("Pré-processamento")
    assume_percent = st.checkbox("Massa já em %", value=False)
    normalize_start = st.checkbox("Normalizar para 100% no início (antes do corte)", value=True)

    st.header("Corte (início da curva)")
    cut_Tmin = st.number_input("Cortar antes de T ≥ (°C)", value=0.0, step=0.1, format="%.1f")
    cut_use = st.checkbox("Ativar corte por temperatura", value=False)
    cut_Nfirst = st.number_input("Remover primeiros N pontos (após T ≥)", min_value=0, value=0, step=1)

    st.header("Ajuste do início em Y")
    y_target_enabled = st.checkbox("Fixar Y inicial em (m%)", value=True)
    y_target_start = st.number_input("Valor alvo para o 1º ponto (m%)", min_value=0.0, value=100.0, step=0.1, format="%.1f")
    y_offset = st.number_input("Offset adicional em Y (m%)", value=0.0, step=0.1, format="%.1f")

    st.header("Suavização")
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

    st.header("Estilo global")
    title_size = st.number_input("Tamanho do título", 6, 48, 16, 1)
    label_size = st.number_input("Tamanho dos rótulos dos eixos", 6, 36, 14, 1)
    tick_size = st.number_input("Tamanho dos ticks", 6, 28, 12, 1)
    legend_size = st.number_input("Tamanho da legenda", 6, 28, 12, 1)

    st.header("Faixas dos eixos")
    if "x_min" not in st.session_state: st.session_state["x_min"] = 0.0
    if "x_max" not in st.session_state: st.session_state["x_max"] = 1000.0
    if "y_min" not in st.session_state: st.session_state["y_min"] = 0.0
    if "y_max" not in st.session_state: st.session_state["y_max"] = 110.0
    x_min = st.number_input("X min (°C)", key="x_min", step=1.0)
    x_max = st.number_input("X max (°C)", key="x_max", step=1.0)
    y_min = st.number_input("Y min (m%)", key="y_min", step=1.0)
    y_max = st.number_input("Y max (m%)", key="y_max", step=1.0)

    st.header("Exportação")
    dpi_export = st.slider("DPI para exportação", min_value=150, max_value=1200, value=600, step=50)

tga_opts = TGAOptions(
    assume_percent=assume_percent,
    normalize_start=normalize_start,
    smoothing="Savitzky-Golay" if (SCIPY_OK and smoothing.startswith("Savitzky-Golay")) else ("MediaMovel" if smoothing=="MediaMovel" else "Nenhum"),
    ma_window=int(ma_window),
    sg_window=int(sg_window),
    sg_poly=int(sg_poly),
)

trim_opts = TrimAlignOptions(
    cut_Tmin=float(cut_Tmin) if cut_use else None,
    cut_Nfirst=int(cut_Nfirst),
    y_target_start=float(y_target_start) if y_target_enabled else None,
    y_offset=float(y_offset),
)

uploaded_files = st.file_uploader("Envie 1 ou mais arquivos .txt/.csv", type=["txt","csv"], accept_multiple_files=True)

if uploaded_files:
    all_processed: Dict[str, pd.DataFrame] = {}
    mapping_per_file: Dict[str, Dict[str,str]] = {}

    st.subheader("Mapeamento de Colunas")
    for f in uploaded_files:
        df_raw, mapping_guess, header_row, sep_used = robust_read_to_df(f.getvalue(), decimal_hint=decimal_hint)
        st.markdown(f"**{f.name}** — cabeçalho na linha {header_row+1} • sep: `{sep_used}`")

        cols = list(df_raw.columns)
        col_temp = st.selectbox(f"Temperatura ({f.name})", cols, index=cols.index(mapping_guess["temperature"]) if "temperature" in mapping_guess else 0, key=f"{f.name}_temp")
        col_mass = st.selectbox(f"Massa (g/mg) ({f.name})", ["(nenhuma)"]+cols, index=(cols.index(mapping_guess["mass"])+1) if "mass" in mapping_guess else 0, key=f"{f.name}_mass")
        col_mpct = st.selectbox(f"Massa % ({f.name})", ["(nenhuma)"]+cols, index=(cols.index(mapping_guess["mass_pct"])+1) if "mass_pct" in mapping_guess else 0, key=f"{f.name}_mpct")

        mapping = {"temperature": col_temp}
        if col_mass != "(nenhuma)": mapping["mass"] = col_mass
        if col_mpct != "(nenhuma)": mapping["mass_pct"] = col_mpct
        mapping_per_file[f.name] = mapping

        try:
            df_proc = process_single(df_raw, mapping, tga_opts, trim_opts)
            all_processed[f.name] = df_proc
        except Exception as e:
            st.error(f"{f.name}: erro — {e}")

    if all_processed:
        st.subheader("Estilo por série e inclusão")
        style_cfg: Dict[str, Dict[str, Optional[str]]] = {}
        include_series: Dict[str, bool] = {}
        for name in all_processed.keys():
            with st.expander(name, expanded=True):
                include = st.checkbox("Incluir na plotagem combinada", value=True, key=f"{name}_include")
                label = st.text_input("Legenda (rótulo)", value=name.rsplit(".",1)[0], key=f"{name}_label")
                color = st.color_picker("Cor da linha", value="#000000", key=f"{name}_color")
                lw = st.number_input("Espessura da linha", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key=f"{name}_lw")
            include_series[name] = include
            style_cfg[name] = {"label": label, "color": color, "lw": lw}


        # ---------- Ajuste rápido de faixas (1 clique) ----------
        def _nice_round(v, base=5.0, mode="floor"):
            if not np.isfinite(v): 
                return v
            import math
            if mode == "floor":
                return base * math.floor(v / base)
            elif mode == "ceil":
                return base * math.ceil(v / base)
            return v

        def _compute_bounds(data_dict, include_dict):
            xs, ys_tga, ys_dtg = [], [], []
            for nm, d in data_dict.items():
                if not include_dict.get(nm, True):
                    continue
                xs.append(d["Temperature"].to_numpy())
                ys_tga.append(d["Mass_pct"].to_numpy())
                ys_dtg.append(d["DTG_(-%/°C)"].to_numpy())
            if xs:
                X = np.concatenate(xs)
            else:
                X = np.array([])
            if ys_tga:
                Yt = np.concatenate(ys_tga)
            else:
                Yt = np.array([])
            if ys_dtg:
                Yd = np.concatenate(ys_dtg)
            else:
                Yd = np.array([])
            return X, Yt, Yd

        st.subheader("Ajuste rápido (X/Y)")
        Xall, Yt_all, Yd_all = _compute_bounds(all_processed, include_series)

        cA, cB, cC, cD, cE = st.columns(5)
        if cA.button("Auto (justo)"):
            if Xall.size:
                xmin, xmax = float(np.nanmin(Xall)), float(np.nanmax(Xall))
                st.session_state["x_min"] = _nice_round(xmin, base=5.0, mode="floor")
                st.session_state["x_max"] = _nice_round(xmax, base=5.0, mode="ceil")
            if Yt_all.size:
                ymin, ymax = float(np.nanmin(Yt_all)), float(np.nanmax(Yt_all))
                st.session_state["y_min"] = _nice_round(ymin, base=1.0, mode="floor")
                st.session_state["y_max"] = _nice_round(ymax, base=1.0, mode="ceil")

        if cB.button("Auto (+10% margem)"):
            if Xall.size:
                xmin, xmax = float(np.nanmin(Xall)), float(np.nanmax(Xall))
                pad = 0.1 * max(1e-9, xmax - xmin)
                st.session_state["x_min"] = _nice_round(xmin - pad, base=5.0, mode="floor")
                st.session_state["x_max"] = _nice_round(xmax + pad, base=5.0, mode="ceil")
            if Yt_all.size:
                ymin, ymax = float(np.nanmin(Yt_all)), float(np.nanmax(Yt_all))
                pad = 0.1 * max(1e-9, ymax - ymin)
                st.session_state["y_min"] = _nice_round(ymin - pad, base=1.0, mode="floor")
                st.session_state["y_max"] = _nice_round(ymax + pad, base=1.0, mode="ceil")

        if cC.button("Quantis 1–99%"):
            if Xall.size:
                xmin, xmax = np.nanquantile(Xall, 0.01), np.nanquantile(Xall, 0.99)
                st.session_state["x_min"] = float(xmin)
                st.session_state["x_max"] = float(xmax)
            if Yt_all.size:
                ymin, ymax = np.nanquantile(Yt_all, 0.01), np.nanquantile(Yt_all, 0.99)
                st.session_state["y_min"] = float(ymin)
                st.session_state["y_max"] = float(ymax)

        if cD.button("Y = 0–100%"):
            st.session_state["y_min"] = 0.0
            st.session_state["y_max"] = 100.0

        if cE.button("Redefinir (padrão)"):
            st.session_state["x_min"] = 0.0
            st.session_state["x_max"] = 1000.0
            st.session_state["y_min"] = 0.0
            st.session_state["y_max"] = 110.0
        
                # ---------- Gráfico TGA ----------
        st.subheader("Gráfico Combinado — TGA")
        fig1, ax1 = plt.subplots()
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            ax1.plot(d["Temperature"], d["Mass_pct"], label=cfg["label"], color=cfg["color"], linewidth=cfg["lw"])
        ax1.set_xlabel("Temperatura (°C)", fontsize=label_size)
        ax1.set_ylabel("Massa (%)", fontsize=label_size)
        ax1.set_title("TGA — Massa (%) vs Temperatura", fontsize=title_size)
        ax1.tick_params(axis='both', labelsize=tick_size)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        leg1 = ax1.legend(fontsize=legend_size)
        st.pyplot(fig1, clear_figure=True)

        # Export buttons
        buf_png1 = io.BytesIO()
        fig1.savefig(buf_png1, format="png", dpi=dpi_export, bbox_inches="tight")
        buf_png1.seek(0)
        buf_svg1 = io.BytesIO()
        fig1.savefig(buf_svg1, format="svg", bbox_inches="tight")
        buf_svg1.seek(0)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Baixar TGA (PNG)", data=buf_png1, file_name="TGA_combinado.png", mime="image/png")
        with c2:
            st.download_button("⬇️ Baixar TGA (SVG vetorial)", data=buf_svg1, file_name="TGA_combinado.svg", mime="image/svg+xml")

        # ---------- Gráfico DTG ----------
        st.subheader("Gráfico Combinado — DTG")
        fig2, ax2 = plt.subplots()
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            ax2.plot(d["Temperature"], d["DTG_(-%/°C)"], label=cfg["label"], color=cfg["color"], linewidth=cfg["lw"])
        ax2.set_xlabel("Temperatura (°C)", fontsize=label_size)
        ax2.set_ylabel("-d(M%)/dT (%/°C)", fontsize=label_size)
        ax2.set_title("DTG — Derivada da Massa (%)", fontsize=title_size)
        ax2.tick_params(axis='both', labelsize=tick_size)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.set_xlim(x_min, x_max)
        leg2 = ax2.legend(fontsize=legend_size)
        st.pyplot(fig2, clear_figure=True)

        buf_png2 = io.BytesIO()
        fig2.savefig(buf_png2, format="png", dpi=dpi_export, bbox_inches="tight")
        buf_png2.seek(0)
        buf_svg2 = io.BytesIO()
        fig2.savefig(buf_svg2, format="svg", bbox_inches="tight")
        buf_svg2.seek(0)
        c3, c4 = st.columns(2)
        with c3:
            st.download_button("⬇️ Baixar DTG (PNG)", data=buf_png2, file_name="DTG_combinado.png", mime="image/png")
        with c4:
            st.download_button("⬇️ Baixar DTG (SVG vetorial)", data=buf_svg2, file_name="DTG_combinado.svg", mime="image/svg+xml")

        st.caption("Dica: use SVG para edição vetorial em softwares como Inkscape/Illustrator; use PNG com DPI alto para publicação.")
else:
    st.info("Envie um ou mais arquivos para visualizar TGA/DTG.")





