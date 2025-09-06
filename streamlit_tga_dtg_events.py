# app.py — Limpar cabeçalho e padronizar nomes (time, temperature, mass, mass_pct)

import re
import io
import os
import unicodedata
import streamlit as st

st.set_page_config(page_title="Limpar Cabeçalho TXT (TGA/DTG)", page_icon="🧹", layout="centered")
st.title("🧹 Limpar Cabeçalho de TXT (com padronização TGA/DTG)")

st.write(
    "Envie um `.txt` com cabeçalho + tabela. O app detecta a tabela, "
    "remove o cabeçalho extra e exporta um TXT com colunas padronizadas: "
    "`time`, `temperature`, `mass`, `mass_pct` (quando aplicável)."
)

# --------------------------
# Helpers
# --------------------------
NUM_RE = re.compile(r'^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$')

def split_tokens(line: str):
    return re.findall(r'\S+', line.strip())

def is_numeric_token(tok: str) -> bool:
    return bool(NUM_RE.match(tok.replace(",", ".")))

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def norm_token(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r'[\[\]\(\)°%]', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    return s

def looks_like_column_header(line: str) -> bool:
    tokens = split_tokens(line)
    if len(tokens) < 2:
        return False
    alpha_tokens = sum(any(ch.isalpha() for ch in t) and not any(ch.isdigit() for ch in t) for t in tokens)
    has_colon = any(':' in t for t in tokens)
    return (alpha_tokens >= 2) and (not has_colon)

def find_table_start(lines):
    step_positions = [i for i, ln in enumerate(lines) if "[step]" in ln.lower()]
    search_starts = step_positions + [0]
    for start in search_starts:
        for i in range(start, len(lines) - 2):
            if looks_like_column_header(lines[i]):
                header_idx = i
                units_idx = i + 1
                data_idx = i + 2
                if data_idx < len(lines) - 1:
                    t1 = split_tokens(lines[data_idx])
                    t2 = split_tokens(lines[data_idx + 1])
                    if len(t1) >= 2 and len(t2) >= 2:
                        nratio1 = sum(is_numeric_token(x) for x in t1) / max(1, len(t1))
                        nratio2 = sum(is_numeric_token(x) for x in t2) / max(1, len(t2))
                        if nratio1 >= 0.6 and nratio2 >= 0.6:
                            return header_idx, units_idx, data_idx
    # fallback
    for i in range(len(lines) - 6):
        t0 = split_tokens(lines[i])
        if len(t0) < 2:
            continue
        nratio0 = sum(is_numeric_token(x) for x in t0) / len(t0)
        if nratio0 < 0.6:
            continue
        num_cols = len(t0)
        good = True
        for k in range(1, 6):
            tk = split_tokens(lines[i + k])
            if len(tk) < 2:
                good = False; break
            nratio = sum(is_numeric_token(x) for x in tk) / len(tk)
            if nratio < 0.6 or abs(len(tk) - num_cols) > 1:
                good = False; break
        if good:
            return None, None, i
    return None, None, None

def build_dataframe_like(lines, idx_header, idx_units, idx_data, max_rows_preview=50):
    col_names = split_tokens(lines[idx_header]) if idx_header is not None else None
    units_line = split_tokens(lines[idx_units]) if idx_units is not None else None

    rows = []
    num_cols_target = None
    for j in range(idx_data, len(lines)):
        ln = lines[j].strip()
        if not ln: break
        if ln.startswith('[') and ln.endswith(']'): break
        toks = split_tokens(ln)
        if len(toks) < 2: continue
        if any(k in ln.lower() for k in (":", "segment", "started", "version", "entry", "log", "calibration")):
            continue
        if num_cols_target is None:
            num_cols_target = len(toks)
        elif abs(len(toks) - num_cols_target) > 1:
            break
        rows.append(toks)
    if not col_names:
        n = max((len(r) for r in rows), default=0)
        col_names = [f"col{i+1}" for i in range(n)]

    # Ajuste de colunas
    ncols = len(col_names)
    clean_rows = []
    for r in rows:
        if len(r) == ncols:
            clean_rows.append(r)
        elif len(r) == ncols - 1:
            clean_rows.append(r + [""])
        elif len(r) > ncols:
            clean_rows.append(r[:ncols])

    return col_names, units_line, clean_rows

def make_txt(col_names, rows, sep, include_header=True, decimal_to_dot=False):
    def fix_decimal(tok: str) -> str:
        return tok.replace(",", ".") if decimal_to_dot else tok
    out = io.StringIO()
    if include_header and col_names:
        out.write(sep.join(col_names) + "\n")
    for r in rows:
        out.write(sep.join(fix_decimal(x) for x in r) + "\n")
    return out.getvalue()

# --- Nova lógica: padronizar cabeçalhos ---
def standardize_header(col_names, units=None, sample_rows=None):
    """
    Tenta mapear para: time, temperature, mass, mass_pct (quando fizer sentido).
    Usa (1) nomes, (2) unidades e (3) heurística numérica dos dados.
    """
    N = len(col_names)
    units = units or [""] * N

    # 1) candidatos por coluna
    candidates = []
    for i in range(N):
        name = norm_token(col_names[i])
        unit = norm_token(units[i])
        cand = None
        # time
        if "time" in name or "tempo" in name or unit in ("s", "min", "ms"):
            cand = "time"
        # temperature
        if cand is None and ("temp" in name or "temperat" in name or "c" == unit or "oc" in unit or "k" == unit):
            cand = "temperature"
        # mass/weight
        if cand is None and (("weight" in name) or ("mass" in name) or ("massa" in name)):
            if "%" in units[i] or "percent" in unit or "pct" in unit:
                cand = "mass_pct"
            elif any(u in unit for u in ("g", "kg", "mg", "ug")):
                cand = "mass"
            else:
                cand = "mass"  # provisório; resolvemos abaixo por heurística
        candidates.append(cand)

    # 2) Heurística para duplicado weight sem unidades (ex.: Weight, Weight)
    #    Se houver duas 'mass', marque a com valores ~100 como mass_pct.
    if sample_rows and candidates.count("mass") >= 2:
        # pega colunas mass
        mass_idxs = [i for i,c in enumerate(candidates) if c == "mass"]
        # média da 1ª linha para cada coluna
        first = sample_rows[0]
        try:
            vals = []
            for idx in mass_idxs:
                v = float(first[idx].replace(",", "."))
                vals.append((idx, v))
            # coluna cujo valor está em [60..120] provavelmente é %
            idx_pct = None
            for idx, v in vals:
                if 60 <= v <= 120:
                    idx_pct = idx; break
            if idx_pct is not None:
                # marca como mass_pct; mantém a outra como mass
                for i in mass_idxs:
                    candidates[i] = "mass_pct" if i == idx_pct else "mass"
        except Exception:
            pass

    # 3) resolve conflitos restantes atribuindo nomes únicos
    used = {}
    out = [""] * N
    for i, cand in enumerate(candidates):
        if cand is None:
            # fallback por posição
            out[i] = norm_token(col_names[i]).replace(" ", "_") or f"col{i+1}"
            continue
        base = cand
        k = 1
        while used.get(cand, False):
            k += 1
            cand = f"{base}_{k}"
        used[cand] = True
        out[i] = cand

    return out

# --------------------------
# UI
# --------------------------
uploaded = st.file_uploader("Envie o arquivo .txt", type=["txt"])

c1, c2 = st.columns(2)
with c1:
    custom_marker = st.text_input("Marcador antes da tabela (opcional)", value="[step]")
with c2:
    output_sep_label = st.selectbox(
        "Delimitador de saída",
        ["Tab (\\t)", "Vírgula (,)", "Ponto e vírgula (;)", "Espaço ( )"],
        index=0,
    )

sep_map = {
    "Tab (\\t)": "\t",
    "Vírgula (,)": ",",
    "Ponto e vírgula (;)": ";",
    "Espaço ( )": " ",
}

include_header = st.checkbox("Incluir linha de cabeçalho na saída", value=True)
decimal_to_dot = st.checkbox("Trocar vírgula por ponto nos decimais", value=False)
manual_skip = st.number_input("Ignorar N linhas manualmente (opcional)", min_value=0, value=0, step=1)

st.divider()
st.subheader("Padronização de cabeçalho")
auto_standardize = st.checkbox("Padronizar automaticamente para time / temperature / mass / mass_pct", value=True)
allow_manual_override = st.checkbox("Permitir sobrescrever via seleção manual", value=True)

if uploaded:
    raw = uploaded.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    lines_all = text.splitlines()

    # recorte por skip/manual
    if manual_skip > 0:
        lines = lines_all[int(manual_skip):]
    else:
        lines = lines_all

    # recorte por marcador
    if custom_marker and custom_marker.strip():
        lower_marker = custom_marker.lower()
        for i, ln in enumerate(lines):
            if lower_marker in ln.lower():
                lines = lines[i:]
                break

    h_idx, u_idx, d_idx = find_table_start(lines)

    if d_idx is None:
        st.error("Não consegui encontrar automaticamente o início da tabela. "
                 "Tente informar o 'Marcador antes da tabela' ou usar 'Ignorar N linhas'.")
    else:
        col_names, units_line, rows = build_dataframe_like(lines, h_idx, u_idx, d_idx)
        if not rows:
            st.warning("Tabela detectada, mas sem linhas válidas. Ajuste o marcador ou 'Ignorar N linhas'.")
        else:
            st.success(f"Tabela detectada! Colunas originais: {len(col_names)} • Linhas (preview): {min(len(rows), 50)}")

            # 1) Padronização automática
            std_cols = col_names[:]
            if auto_standardize:
                std_cols = standardize_header(col_names, units=units_line, sample_rows=rows[:3])

            # 2) Sobrescrita manual (opcional)
            final_cols = std_cols[:]
            if allow_manual_override:
                st.caption("Se necessário, selecione quais colunas correspondem a cada papel:")
                opts = [f"{i+1}: {col_names[i]} → {std_cols[i]}" for i in range(len(col_names))]
                def idx_from_opt(opt):
                    return int(opt.split(":")[0]) - 1

                time_choice = st.selectbox("Coluna para `time`", ["(não alterar)"] + opts, index=0)
                temp_choice = st.selectbox("Coluna para `temperature`", ["(não alterar)"] + opts, index=0)
                mass_choice = st.selectbox("Coluna para `mass`", ["(não alterar)"] + opts, index=0)
                masspct_choice = st.selectbox("Coluna para `mass_pct`", ["(não alterar)"] + opts, index=0)

                # aplica sobrescritas
                chosen = {
                    "time": time_choice,
                    "temperature": temp_choice,
                    "mass": mass_choice,
                    "mass_pct": masspct_choice,
                }
                for role, choice in chosen.items():
                    if choice != "(não alterar)":
                        j = idx_from_opt(choice)
                        final_cols[j] = role

            # Preview
            preview_txt = make_txt(final_cols, rows[:50], sep="\t", include_header=True, decimal_to_dot=decimal_to_dot)
            st.text_area("Prévia (primeiras linhas)", preview_txt, height=240)

            # Exporta
            out_txt = make_txt(final_cols, rows, sep=sep_map[output_sep_label],
                               include_header=include_header, decimal_to_dot=decimal_to_dot)
            base_name = os.path.splitext(uploaded.name)[0]
            out_name = f"{base_name}_limpo.txt"
            st.download_button("⬇️ Baixar TXT limpo (padronizado)", data=out_txt.encode("utf-8"),
                               file_name=out_name, mime="text/plain")

st.caption("Dica: se vierem duas colunas 'Weight', a primeira tende a ser massa absoluta (`mass`) e a segunda massa percentual (`mass_pct`).")

