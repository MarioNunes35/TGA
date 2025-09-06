
# streamlit_tga_dtg_events.py (hardened)
import io
import re
import numpy as np
import pandas as pd
import streamlit as st

# Plotly optional
PLOTLY_OK = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False

# Matplotlib fallback
import matplotlib.pyplot as plt

st.set_page_config(page_title="TGA/DTG Viewer", layout="wide")

# ------------------------ Helpers robustos ------------------------
def _coerce_float(val, default=None):
    try:
        if val is None:
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return float(val)
    except Exception:
        return default

def _coerce_int(val, default=None):
    try:
        if val is None:
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return int(val)
    except Exception:
        return default

def _safe_font_size(x, default):
    v = _coerce_int(x, default)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        v = default
    if v <= 0:
        v = default
    return int(v)

def _sanitized_range(vmin, vmax):
    vmin = _coerce_float(vmin, None)
    vmax = _coerce_float(vmax, None)
    if vmin is None or vmax is None:
        return None
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    # swap if reversed
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    # expand tiny/zero span
    if vmin == vmax:
        eps = 1.0 if vmin == 0 else abs(vmin) * 1e-6 + 1e-6
        return [float(vmin - eps), float(vmax + eps)]
    return [float(vmin), float(vmax)]

def _normalize_and_automap(df: pd.DataFrame):
    cols = [str(c).strip() for c in df.columns]
    seen, new_cols = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    guess = {}

    # Temperatura (heurística simples)
    for cand in ["Temperature", "Temp", "T", "temperatura", "Temperature (°C)"]:
        if cand in df.columns:
            guess["temperature"] = cand
            break

    # Se existirem duas "Weight", decidir qual é massa (%) e qual é massa absoluta
    wcols = [c for c in df.columns if c.lower() == "weight" or c.lower().startswith("weight_")]
    if len(wcols) >= 2:
        c1, c2 = wcols[:2]

        def first_float(col):
            try:
                return float(str(df[col].dropna().iloc[0]).replace(",", "."))
            except Exception:
                return None

        v1, v2 = first_float(c1), first_float(c2)

        if v1 is not None and v2 is not None:
            if v1 > 50 and v2 < 50:
                pct, mass = c1, c2
            elif v2 > 50 and v1 < 50:
                pct, mass = c2, c1
            else:
                try:
                    mx1 = pd.to_numeric(df[c1], errors="coerce").max()
                    mx2 = pd.to_numeric(df[c2], errors="coerce").max()
                    pct, mass = (c1, c2) if (mx1 if np.isfinite(mx1) else -np.inf) >= (mx2 if np.isfinite(mx2) else -np.inf) else (c2, c1)
                except Exception:
                    pct, mass = c1, c2
        else:
            pct, mass = c1, c2

        df.rename(columns={mass: "Mass", pct: "Mass_pct"}, inplace=True)
        guess["mass"] = "Mass"
        guess["mass_pct"] = "Mass_pct"
    else:
        # tentativas simples
        for cand in ["Mass", "Weight", "Massa", "mass (mg)", "mass (g)"]:
            if cand in df.columns:
                guess["mass"] = cand
                break
        for cand in ["Mass_pct", "Mass %", "Massa %", "Weight %", "mass%"]:
            if cand in df.columns:
                guess["mass_pct"] = cand
                break

    return df, guess

def apply_plotly_layout(fig, title_text, ytitle, title_size, label_size, tick_size, legend_size):
    layout = dict(
        template="plotly_white",
        xaxis_title="Temperatura (°C)",
        yaxis_title=ytitle,
        title_text=title_text,
        title_font_size=_safe_font_size(title_size, 16),
    )
    xaxis = dict(
        tickfont=dict(size=_safe_font_size(tick_size, 12)),
        titlefont=dict(size=_safe_font_size(label_size, 14)),
    )
    yaxis = dict(
        tickfont=dict(size=_safe_font_size(tick_size, 12)),
        titlefont=dict(size=_safe_font_size(label_size, 14)),
    )

    x_rng = _sanitized_range(st.session_state.get("x_min"), st.session_state.get("x_max"))
    y_rng = _sanitized_range(st.session_state.get("y_min"), st.session_state.get("y_max"))
    if x_rng is not None:
        xaxis["range"] = x_rng
    if y_rng is not None:
        yaxis["range"] = y_rng

    layout["xaxis"] = xaxis
    layout["yaxis"] = yaxis

    leg = _safe_font_size(legend_size, None)
    if leg is not None:
        layout["legend"] = dict(font=dict(size=leg))

    fig.update_layout(**layout)

# ------------------------ Leitura robusta ------------------------
def robust_read_to_df(content_bytes: bytes, decimal_hint: str = "."):
    text = content_bytes.decode("utf-8", errors="replace")
    decimal = "," if decimal_hint == "," else "."
    sep_used = r"\s+"
    header_row = 0

    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", decimal=decimal)
        return df, {}, header_row, sep_used
    except Exception:
        pass

    try:
        df = pd.read_csv(io.StringIO(text), sep=",", engine="python", decimal=decimal)
        sep_used = ","
        return df, {}, header_row, sep_used
    except Exception:
        pass

    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", decimal=decimal)
    sep_used = ";"
    return df, {}, header_row, sep_used

# ------------------------ Processamento ------------------------
def process_single(df_raw: pd.DataFrame, mapping: dict, baseline_mode: str = "first"):
    t_col = mapping["temperature"]
    T = pd.to_numeric(df_raw[t_col].astype(str).str.replace(",", "."), errors="coerce")

    mpct = None
    if "mass_pct" in mapping and mapping["mass_pct"] in df_raw.columns:
        mpct = pd.to_numeric(df_raw[mapping["mass_pct"]].astype(str).str.replace(",", "."), errors="coerce")

    if mpct is None and "mass" in mapping and mapping["mass"] in df_raw.columns:
        mabs = pd.to_numeric(df_raw[mapping["mass"]].astype(str).str.replace(",", "."), errors="coerce")
        base = None
        if baseline_mode == "first":
            mnotna = mabs.dropna()
            base = mnotna.iloc[0] if len(mnotna) else None
        else:
            try:
                base = np.nanmax(mabs.values)
            except Exception:
                base = None
        if base is not None and base != 0 and np.isfinite(base):
            mpct = (mabs / base) * 100.0

    if mpct is None:
        raise ValueError("Coluna de massa/porcentagem não encontrada. Ajuste o mapeamento.")

    with np.errstate(invalid="ignore", divide="ignore"):
        dt = np.gradient(T)
        dM = np.gradient(mpct)
        dtg = -dM / dt

    out = pd.DataFrame({
        "Temperature": T,
        "Mass_pct": mpct,
        "DTG_(-%/°C)": dtg
    })
    out = out.dropna(subset=["Temperature", "Mass_pct"])
    return out

# ------------------------ UI ------------------------
st.title("TGA / DTG — Leitor e Plotador")

with st.sidebar:
    st.header("Preferências")
    decimal_hint = st.selectbox("Separador decimal nos arquivos", [".", ","], index=0)
    baseline_mode = st.radio("Se só houver Massa, calcular % usando:", ["first", "max"], index=0)

    st.markdown("---")
    st.subheader("Estilo dos gráficos")
    title_size = st.number_input("Tamanho do título", min_value=6, max_value=48, value=16, step=1)
    label_size = st.number_input("Tamanho dos rótulos dos eixos", min_value=6, max_value=36, value=14, step=1)
    tick_size = st.number_input("Tamanho dos ticks", min_value=6, max_value=28, value=12, step=1)
    legend_size = st.number_input("Tamanho da legenda", min_value=6, max_value=28, value=12, step=1)

    st.markdown("---")
    st.subheader("Limites dos eixos (inicial)")
    if "x_min" not in st.session_state: st.session_state["x_min"] = 0.0
    if "x_max" not in st.session_state: st.session_state["x_max"] = 1000.0
    if "y_min" not in st.session_state: st.session_state["y_min"] = 0.0
    if "y_max" not in st.session_state: st.session_state["y_max"] = 110.0

    st.session_state["x_min"] = st.number_input("x_min (°C)", value=float(st.session_state["x_min"]))
    st.session_state["x_max"] = st.number_input("x_max (°C)", value=float(st.session_state["x_max"]))
    st.session_state["y_min"] = st.number_input("y_min (%)", value=float(st.session_state["y_min"]))
    st.session_state["y_max"] = st.number_input("y_max (%)", value=float(st.session_state["y_max"]))

uploaded_files = st.file_uploader(
    "Envie 1 ou mais arquivos .txt/.csv", type=["txt", "csv"], accept_multiple_files=True
)

if uploaded_files:
    all_processed = {}
    mapping_per_file = {}

    st.subheader("Mapeamento de Colunas")
    for f in uploaded_files:
        df_raw, mapping_guess, header_row, sep_used = robust_read_to_df(
            f.getvalue(), decimal_hint=decimal_hint
        )
        st.markdown(f"**{f.name}** — cabeçalho na linha {header_row+1} • sep: `{sep_used}`")

        df_raw, guess2 = _normalize_and_automap(df_raw)
        if isinstance(mapping_guess, dict):
            mapping_guess.update({k: v for k, v in guess2.items() if v})

        cols = list(df_raw.columns)

        t_default = cols.index(mapping_guess.get("temperature", cols[0])) if cols else 0
        col_temp = st.selectbox(
            f"Temperatura ({f.name})",
            cols, index=t_default,
            key=f"{f.name}_temp"
        )

        mass_options = ["(nenhuma)"] + cols
        m_default = mass_options.index(mapping_guess.get("mass")) if mapping_guess.get("mass") in cols else 0
        col_mass = st.selectbox(
            f"Massa (g/mg) ({f.name})",
            mass_options, index=m_default,
            key=f"{f.name}_mass"
        )

        mpct_options = ["(nenhuma)"] + cols
        mp_default = mpct_options.index(mapping_guess.get("mass_pct")) if mapping_guess.get("mass_pct") in cols else 0
        col_mpct = st.selectbox(
            f"Massa % ({f.name})",
            mpct_options, index=mp_default,
            key=f"{f.name}_mpct"
        )

        mapping = {"temperature": col_temp}
        if col_mass != "(nenhuma)":
            mapping["mass"] = col_mass
        if col_mpct != "(nenhuma)":
            mapping["mass_pct"] = col_mpct
        mapping_per_file[f.name] = mapping

        try:
            df_proc = process_single(df_raw, mapping, baseline_mode=baseline_mode)
            all_processed[f.name] = df_proc
        except Exception as e:
            st.error(f"{f.name}: erro — {e}")

    if not all_processed:
        st.stop()

    st.subheader("Estilo por Série")
    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    style_cfg = {}
    include_series = {}
    for i, (name, df_) in enumerate(all_processed.items()):
        with st.expander(f"Série: {name}", expanded=False):
            include_series[name] = st.checkbox("Incluir no gráfico", value=True, key=f"incl_{name}")
            color = st.color_picker("Cor", value=default_colors[i % len(default_colors)], key=f"color_{name}")
            lw = st.number_input("Espessura da linha", min_value=0.5, max_value=8.0, value=2.0, step=0.5, key=f"lw_{name}")
            label = st.text_input("Rótulo (legenda)", value=name, key=f"label_{name}")
            style_cfg[name] = {"color": color, "lw": lw, "label": label}

            st.download_button(
                "Baixar CSV processado",
                data=df_.to_csv(index=False).encode("utf-8"),
                file_name=f"{name}_processed.csv",
                mime="text/csv",
                key=f"dl_{name}"
            )

    # --------- Gráfico TGA ---------
    st.subheader("Gráfico Combinado — TGA")
    if PLOTLY_OK:
        fig1 = go.Figure()
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            fig1.add_trace(go.Scatter(
                x=d["Temperature"], y=d["Mass_pct"], mode="lines",
                name=cfg["label"],
                line=dict(color=cfg["color"], width=float(cfg["lw"]))
            ))
        apply_plotly_layout(
            fig1,
            title_text="TGA — Massa (%) vs Temperatura",
            ytitle="Massa (%)",
            title_size=title_size, label_size=label_size, tick_size=tick_size, legend_size=legend_size
        )
        st.plotly_chart(fig1, use_container_width=True, config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": []
        })
    else:
        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=110)
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            ax1.plot(d["Temperature"], d["Mass_pct"], label=cfg["label"], linewidth=float(cfg["lw"]))
        ax1.set_xlabel("Temperatura (°C)", fontsize=_safe_font_size(14, 14))
        ax1.set_ylabel("Massa (%)", fontsize=_safe_font_size(14, 14))
        ax1.set_title("TGA — Massa (%) vs Temperatura", fontsize=_safe_font_size(16, 16))
        ax1.tick_params(axis="both", labelsize=_safe_font_size(12, 12))
        xr = _sanitized_range(st.session_state.get("x_min"), st.session_state.get("x_max"))
        yr = _sanitized_range(st.session_state.get("y_min"), st.session_state.get("y_max"))
        if xr: ax1.set_xlim(*xr)
        if yr: ax1.set_ylim(*yr)
        ax1.legend(fontsize=_safe_font_size(12, 12))
        st.pyplot(fig1, clear_figure=True)

    # --------- Gráfico DTG ---------
    st.subheader("Gráfico Combinado — DTG")
    if PLOTLY_OK:
        fig2 = go.Figure()
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            fig2.add_trace(go.Scatter(
                x=d["Temperature"], y=d["DTG_(-%/°C)"], mode="lines",
                name=cfg["label"],
                line=dict(color=cfg["color"], width=float(cfg["lw"]))
            ))
        apply_plotly_layout(
            fig2,
            title_text="DTG — Derivada da Massa (%)",
            ytitle="-d(M%)/dT (%/°C)",
            title_size=title_size, label_size=label_size, tick_size=tick_size, legend_size=legend_size
        )
        st.plotly_chart(fig2, use_container_width=True, config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": []
        })
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=110)
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            ax2.plot(d["Temperature"], d["DTG_(-%/°C)"], label=cfg["label"], linewidth=float(cfg["lw"]))
        ax2.set_xlabel("Temperatura (°C)", fontsize=_safe_font_size(14, 14))
        ax2.set_ylabel("-d(M%)/dT (%/°C)", fontsize=_safe_font_size(14, 14))
        ax2.set_title("DTG — Derivada da Massa (%)", fontsize=_safe_font_size(16, 16))
        ax2.tick_params(axis="both", labelsize=_safe_font_size(12, 12))
        xr = _sanitized_range(st.session_state.get("x_min"), st.session_state.get("x_max"))
        if xr: ax2.set_xlim(*xr)
        ax2.legend(fontsize=_safe_font_size(12, 12))
        st.pyplot(fig2, clear_figure=True)

    # --------- Exportação em Lote ---------
    st.subheader("Exportação em Lote")
    import zipfile, os, tempfile
    with tempfile.TemporaryDirectory() as td:
        for name, d in all_processed.items():
            d.to_csv(os.path.join(td, f"{name}_processed.csv"), index=False)
        zip_path = os.path.join(td, "tga_dtg_processed.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in all_processed.keys():
                zf.write(os.path.join(td, f"{name}_processed.csv"), arcname=f"{name}_processed.csv")
        with open(zip_path, "rb") as fh:
            st.download_button(
                "Baixar todos os CSVs (ZIP)",
                data=fh.read(),
                file_name="tga_dtg_processed.zip",
                mime="application/zip"
            )

























