import pandas as pd
import numpy as np

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normaliza nomes (lowercase, _ no lugar de espaço, remove símbolos comuns)
    def norm(c):
        c = c.strip().lower()
        c = c.replace(' ', '_').replace('%', 'pct').replace('(°c)', '').replace('(mg)', '')
        c = c.replace('.', '_')
        return c

    df.columns = [norm(c) for c in df.columns]

    # mapeia sinônimos de temperatura
    for cand in ['temperature', 'temp', 't']:
        if cand in df.columns:
            df.rename(columns={cand: 'temperature'}, inplace=True)
            break

    # detecta colunas de massa/peso (podem vir duplicadas: weight, weight_1 etc.)
    weight_cols = [c for c in df.columns if c.startswith('weight') or c in ['mass', 'massa', 'sample_weight']]
    # Se houver duas "weight", escolhe a maior (≈100) como porcentagem e a menor como massa absoluta
    if len(weight_cols) >= 2:
        series = []
        for c in weight_cols:
            s = pd.to_numeric(df[c], errors='coerce')
            series.append((c, np.nanmedian(s)))
        # maior mediana → %peso; menor → massa absoluta
        pct_col = max(series, key=lambda x: x[1])[0]
        mass_col = min(series, key=lambda x: x[1])[0]
        ren = {pct_col: 'mass_pct', mass_col: 'mass'}
        df.rename(columns=ren, inplace=True)
    elif len(weight_cols) == 1:
        c = weight_cols[0]
        s = pd.to_numeric(df[c], errors='coerce')
        df.rename(columns={c: 'mass_pct' if np.nanmedian(s) > 50 else 'mass'}, inplace=True)

    # opcional: padroniza time
    if 'time' in df.columns:
        df.rename(columns={'time': 'time_s'}, inplace=True)

    return df


