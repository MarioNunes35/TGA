# streamlit_tga_dtg_events.py
# ------------------------------------------------------------
# App TGA/DTG com mapeamento de colunas e gráficos robustos
# ------------------------------------------------------------

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (opcional; se não houver, cai para Matplotlib)
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
    """Converte para float com segurança (None ou '' -> default)."""
    try:
        if val is None:
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return float(val)
    except Exception:
        return default


def _coerce_int(val, default=None):
    """Converte para int com segurança (None ou '' -> default)."""
    try:
        if val is None:
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return int(val)
    except Exception:
        return default


def _normalize_and_automap(df: pd.DataFrame):
    """
    1) Deduplica nomes de colunas preservando ordem (ex.: Weight -> Weight_1, ...)
    2) Tenta automapear temperatura e massa/massa% quando possível (duas 'Weight').
    Retorna (df_normalizado, mapping_guess_parcial)
    """
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
                # fallback: coluna com maior máximo costuma ser % (inicia perto de 100)
                pct, mass = (c1, c2) if pd.to_numeric(df[c1], errors="coerce").max() >= pd.to_numeric(df[c2], errors="coerce").max() else (c2, c1)
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
    """
    Aplica layout de modo robusto ao Plotly (sem ValueError quando inputs estão vazios).
    """
    layout = dict(
        template="plotly_white",
        xaxis_title="Temperatura (°C)",
        yaxis_title=ytitle,
        title_text=title_text,
        title_font_size=_coerce_int(title_size, 16),
    )
    xaxis = dict(
        tickfont=dict(size=_coerce_int(tick_size, 12)),
        titlefont=dict(size=_coerce_int(label_size, 14)),
    )
    yaxis = dict(
        tickfont=dict(size=_coerce_int(tick_size, 12)),
        titlefont=dict(size=_coerce_int(label_size, 14)),
    )
    x_min = _coerce_float(st.session_state.get("x_min"))
    x_max = _coerce_float(st.session_state.get("x_max"))
    y_min = _coerce_float(st.session_state.get("y_min"))
    y_max = _coerce_float(st.session_state.get("y_max"))
    if x_min is not None and x_max is not None:
        xaxis["range"] = [x_min, x_max]
    if y_min is not None and y_max is not None:
        yaxis["range"] = [y_min, y_max]
    layout["xaxis"] = xaxis
    layout["yaxis"] = yaxis

    leg = _coerce_int(legend_size, None)
    if leg is not None:
        layout["legend"] = dict(font=dict(size=leg))

    fig.update_layout(**layout)


# ------------------------ Leitura robusta ------------------------

def robust_read_to_df(content_bytes: bytes, decimal_hint: str = "."):
    """
    Lê TXT/CSV com separação por espaços múltiplos (\\s+) ou vírgulas/pontos-e-vírgulas.
    Retorna: df, mapping_guess (vazio aqui), header_row(0), sep_used(str)
    """
    text = content_bytes.decode("utf-8", errors="replace")
    decimal = "," if decimal_hint == "," else "."
    sep_used = r"\s+"
    header_row = 0

    # Tentativa 1: whitespace (mais comum em exportações de TGA)
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", decimal=decimal)
        return df, {}, header_row, sep_used
    except Exception:
        pass

    # Tentativa 2: vírgula
    try:
        df = pd.read_csv(io.StringIO(text), sep=",", engine="python", decimal=decimal)
        sep_used = ","
        return df, {}, header_row, sep_used
    except Exception:
        pass

    # Tentativa 3: ponto-e-vírgula
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", decimal=decimal)
    sep_used = ";"
    return df, {}, header_row, sep_used


# ------------------------ Processamento de um arquivo ------------------------

def process_single(df_raw: pd.DataFrame, mapping: dict, baseline_mode: str = "first"):
    """
    Normaliza e retorna DataFrame processado com colunas:
      - Temperature (float)
      - Mass_pct (float)
      - DTG_(-%/°C) (float)  -> derivada negativa de Mass_pct vs Temperature
    """
    # Coluna de temperatura
    t_col = mapping["temperature"]
    T = pd.to_numeric(df_raw[t_col].astype(str).str.replace(",", "."), errors="coerce")

    # Massa % preferencialmente; senão, calcular a partir da massa absoluta
    mpct = None
    if "mass_pct" in mapping:
        mcol = mapping["mass_pct"]
        if mcol in df_raw.columns:
            mpct = pd.to_numeric(df_raw[mcol].astype(str).str.replace(",", "."), errors="coerce")

    if mpct is None and "mass" in mapping and mapping["mass"] in df_raw.columns:
        mabs = pd.to_numeric(df_raw[mapping["mass"]].astype(str).str.replace(",", "."), errors="coerce")
        if baseline_mode == "first":
            base = mabs.dropna().iloc[0] if len(mabs.dropna()) else np.nan
        else:
            base = np.nanmax(mabs.values)
        mpct = (mabs / base) * 100.0 if base and base != 0 else np.nan

    # Se ainda assim não houver Mass_pct, aborta
    if mpct is None:
        raise ValueError("Coluna de massa/porcentagem não encontrada. Ajuste o mapeamento.")

    # Cálculo do DTG: - d(M%)/dT
    # Usamos gradient com T (em °C)
    with np.errstate(invalid="ignore"):
        dt = np.gradient(T)
        dM = np.gradient(mpct)
        dtg = -dM / dt

    out = pd.DataFrame({
        "Temperature": T,
        "Mass_pct": mpct,
        "DTG_(-%/°C)": dtg
    })
    out = out.dropna(subset=["Temperature", "Mass_pct"])  # T e M% são essenciais
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
    "Envie 1 ou mais arquivos .txt/.csv",
    type=["txt", "csv"],
    accept_multiple_files=True
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

        # Normaliza duplicatas e tenta automap
        df_raw, guess2 = _normalize_and_automap(df_raw)
        if isinstance(mapping_guess, dict):
            mapping_guess.update({k: v for k, v in guess2.items() if v})

        cols = list(df_raw.columns)

        # Selectboxes (temperatura obrigatória; massa/massa% opcionais)
        # Use valores guessed quando possível
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

        # Processar
        try:
            df_proc = process_single(df_raw, mapping, baseline_mode=baseline_mode)
            all_processed[f.name] = df_proc
        except Exception as e:
            st.error(f"{f.name}: erro — {e}")

    if not all_processed:
        st.stop()

    # --------- Configuração de estilo por série ----------
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

            # Download CSV processado (individual)
            st.download_button(
                "Baixar CSV processado",
                data=df_.to_csv(index=False).encode("utf-8"),
                file_name=f"{name}_processed.csv",
                mime="text/csv",
                key=f"dl_{name}"
            )

    # --------- Gráfico TGA ----------
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
            "modeBarButtonsToRemove": []  # mantém câmera, zoom, pan, +/-, autoscale, home
        })
    else:
        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=110)
        for name, d in all_processed.items():
            if not include_series.get(name, True):
                continue
            cfg = style_cfg[name]
            ax1.plot(d["Temperature"], d["Mass_pct"], label=cfg["label"], linewidth=float(cfg["lw"]))
        ax1.set_xlabel("Temperatura (°C)", fontsize=label_size)
        ax1.set_ylabel("Massa (%)", fontsize=label_size)
        ax1.set_title("TGA — Massa (%) vs Temperatura", fontsize=title_size)
        ax1.tick_params(axis="both", labelsize=tick_size)
        ax1.set_xlim(st.session_state["x_min"], st.session_state["x_max"])
        ax1.set_ylim(st.session_state["y_min"], st.session_state["y_max"])
        ax1.legend(fontsize=legend_size)
        st.pyplot(fig1, clear_figure=True)

    # --------- Gráfico DTG ----------
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
        ax2.set_xlabel("Temperatura (°C)", fontsize=label_size)
        ax2.set_ylabel("-d(M%)/dT (%/°C)", fontsize=label_size)
        ax2.set_title("DTG — Derivada da Massa (%)", fontsize=title_size)
        ax2.tick_params(axis="both", labelsize=tick_size)
        ax2.set_xlim(st.session_state["x_min"], st.session_state["x_max"])
        ax2.legend(fontsize=legend_size)
        st.pyplot(fig2, clear_figure=True)

    # --------- Exportar todos os processados (ZIP simples em CSVs) ----------
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
























