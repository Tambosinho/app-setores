# app.py
import re
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Company Sector Browser", page_icon="📊", layout="wide")
st.title("📊 Company Sector Browser")
st.caption("Filter by CNAE or by your proprietary 3-level system. Each tab shows a unified table with both namings.")

# ---------------- Constants (fixed columns) ----------------
REQ_COLS = [
    "cnpj","input_company",
    "setor","subsetor","segmento",
    "cnae_primario","cnae_descricao",
    "cnae_secundarios","cnae_secundarios_norm","cnae_secundarios_descricao","cnae_secundarios_pairs",
    "cnae7",
    "setor_progest","subsetor_progest","segmento_progest",
    # optional but present in your example:
    "source","confidence",
]

# ---------------- Helpers ----------------
def format_cnpj(raw: str) -> str:
    """Format a 14-digit CNPJ as 00.000.000/0000-00."""
    if not isinstance(raw, str):
        return ""
    s = re.sub(r"\D", "", raw).zfill(14)
    if len(s) != 14:
        return raw or ""
    return f"{s[0:2]}.{s[2:5]}.{s[5:8]}/{s[8:12]}-{s[12:14]}"

_SPLIT_RE = re.compile(r"[,\|;]+")

def split_list(x: str) -> List[str]:
    """Split a multi-item string like '0210107 | 0220901, 0230600' into a clean list."""
    if not isinstance(x, str) or not x.strip():
        return []
    return [t.strip() for t in _SPLIT_RE.split(x) if t.strip()]

def split_sec_descriptions(row: pd.Series) -> List[str]:
    """
    Build list of secondary CNAE descriptions.
    Prefer `cnae_secundarios_descricao`; if empty, derive from `cnae_secundarios_pairs` by taking text after '—'.
    """
    if isinstance(row.get("cnae_secundarios_descricao"), str) and row["cnae_secundarios_descricao"].strip():
        return split_list(row["cnae_secundarios_descricao"])
    pairs = row.get("cnae_secundarios_pairs")
    out: List[str] = []
    if isinstance(pairs, str) and pairs.strip():
        for it in split_list(pairs):
            # pairs look like "6499999 — Outras atividades ..."
            parts = [p.strip() for p in it.split("—", 1)]
            if len(parts) == 2:
                out.append(parts[1])
            else:
                out.append(it.strip())
    return [x for x in out if x]

def unique_non_null(series: pd.Series) -> list:
    s = series.dropna().astype("string").str.strip()
    s = s[s != ""]
    return sorted(s.unique().tolist())

@st.cache_data(show_spinner=False)
def load_csv() -> pd.DataFrame:
    default = Path("industry_from_rfb_progest.csv")
    if not default.exists():
        st.error("CSV 'industry_from_rfb_progest.csv' not found in the working directory.")
        st.stop()

    df = pd.read_csv(default, dtype="string")

    # Ensure required columns exist (kept strict & simple)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {', '.join(missing)}")
        st.stop()

    # Normalize conveniences
    df = df.copy()
    df["CNPJ_fmt"] = df["cnpj"].map(format_cnpj)
    df["__sec_code_list__"] = df["cnae_secundarios_norm"].map(split_list)
    df["__sec_desc_list__"] = df.apply(split_sec_descriptions, axis=1)

    return df

def common_text_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    st.subheader("🔎 Search")
    c1, c2 = st.columns([2,1])
    with c1:
        name_q = st.text_input("Company name contains", value="", key=f"{key_prefix}_name").strip()
    with c2:
        cnpj_q = st.text_input("CNPJ digits contain", value="", key=f"{key_prefix}_cnpj").strip()

    out = df
    if name_q:
        out = out[out["input_company"].fillna("").str.contains(name_q, case=False, na=False)]
    if cnpj_q:
        digits = re.sub(r"\D", "", cnpj_q)
        if digits:
            out = out[out["cnpj"].fillna("").str.contains(digits, na=False)]
    return out

def unified_results_table(df_filtered: pd.DataFrame, *, widget_key: str):
    # Show both systems together + company & CNPJ (highlight primary & secondary descriptions earlier)
    cols = [
        "input_company","CNPJ_fmt","cnpj",
        "cnae_descricao",  # highlight primary description early
        "cnae_secundarios_descricao",  # highlight secondary description early
        "cnae_primario",
        "setor","subsetor","segmento",
        "cnae_secundarios_norm","cnae_secundarios_pairs",
        "setor_progest","subsetor_progest","segmento_progest",
        "cnae7","source","confidence",
    ]
    present = [c for c in cols if c in df_filtered.columns]
    view = df_filtered[present].rename(columns={
        "input_company": "Empresa",
        "CNPJ_fmt": "CNPJ (formatado)",
        "cnpj": "CNPJ (bruto)",
        "setor": "CNAE Setor",
        "subsetor": "CNAE Subsetor",
        "segmento": "CNAE Segmento",
        "cnae_primario": "CNAE Primário (código)",
        "cnae_descricao": "CNAE Primário (descrição)",
        "cnae_secundarios_norm": "CNAE Secundários (códigos)",
        "cnae_secundarios_descricao": "CNAE Secundários (descrições)",
        "cnae_secundarios_pairs": "CNAE Secundários (pares código — descrição)",
        "setor_progest": "Prop. Setor",
        "subsetor_progest": "Prop. Subsetor",
        "segmento_progest": "Prop. Segmento",
        "cnae7": "CNAE7",
        "source": "Fonte",
        "confidence": "Confiança",
    })
    st.caption(f"Showing **{len(view):,}** rows.")
    st.dataframe(view, use_container_width=True, hide_index=True, key=f"df_{widget_key}")

    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name=f"filtered_companies_{widget_key}.csv",
        mime="text/csv",
        key=f"download_{widget_key}",
    )

# ---------------- Data ----------------
df = load_csv()

# ---------------- Tabs ----------------
tab_cnae, tab_prop = st.tabs(["🏷️ CNAE filters", "🏷️ JP filters (proprietary)"])

# ============================ CNAE TAB ============================
with tab_cnae:
    st.subheader("Filter by CNAE")

    # 1) Text search first so dropdown options reflect only matching companies
    df_base = common_text_filters(df, key_prefix="cnae")

    # Progressive filtering & dependent options (non-null only)

    # --- Emphasize descriptions first (most important) ---
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        # Primary description
        opts_primary_desc = unique_non_null(df_base["cnae_descricao"])
        sel_primary_desc = st.multiselect(
            "⭐ CNAE Primário (descrição)",
            options=opts_primary_desc,
            default=[],
            help="Most used filter."
        )
    df_step = df_base
    if sel_primary_desc:
        df_step = df_step[df_step["cnae_descricao"].isin(sel_primary_desc)]

    with c2:
        # Secondary description list options depend on primary-desc selection
        all_sec_descs = sorted({d for row in df_step["__sec_desc_list__"] for d in (row or [])})
        sel_sec_desc = st.multiselect(
            "⭐ CNAE Secundário (descrição)",
            options=all_sec_descs,
            default=[],
            help="Also a key filter."
        )
        sec_desc_mode = st.radio(
            "Secondary (descr.) match",
            options=["ANY","ALL"],
            index=0,
            horizontal=True,
            key="sec_desc_mode",
            help="ANY = at least one selected description appears. ALL = all must appear."
        )

    # Apply secondary description filter
    if sel_sec_desc:
        if sec_desc_mode == "ANY":
            df_step = df_step[df_step["__sec_desc_list__"].map(lambda L: bool(set(L) & set(sel_sec_desc)))]
        else:
            df_step = df_step[df_step["__sec_desc_list__"].map(lambda L: set(sel_sec_desc).issubset(set(L)))]

    with c3:
        # Primary code options depend on previous selections
        opts_primary_code = unique_non_null(df_step["cnae_primario"])
        sel_cnae_code = st.multiselect(
            "CNAE Primário (código)",
            options=opts_primary_code,
            default=[],
        )

    if sel_cnae_code:
        df_step = df_step[df_step["cnae_primario"].isin(sel_cnae_code)]

    # --- Secondary codes (depend on prior selections) ---
    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        all_sec_codes = sorted({code for row in df_step["__sec_code_list__"] for code in (row or [])})
        sel_sec_codes = st.multiselect(
            "CNAE Secundários (códigos)",
            options=all_sec_codes,
            default=[],
            help="Filters rows that match the selected secondary codes."
        )
    with c5:
        sec_code_mode = st.radio(
            "Secondary (cód.) match",
            options=["ANY","ALL"],
            index=0,
            horizontal=True,
            key="sec_code_mode",
            help="ANY = at least one selected code appears. ALL = all must appear."
        )
    with c6:
        # CNAE hierarchy fields depend on everything chosen so far
        opts_setor = unique_non_null(df_step["setor"])
        sel_setor = st.multiselect("CNAE Setor", options=opts_setor, default=[])

    # Apply secondary codes filter
    if sel_sec_codes:
        if sec_code_mode == "ANY":
            df_step = df_step[df_step["__sec_code_list__"].map(lambda L: bool(set(L) & set(sel_sec_codes)))]
        else:
            df_step = df_step[df_step["__sec_code_list__"].map(lambda L: set(sel_sec_codes).issubset(set(L)))]

    # Further dependent options for subsetor/segmento after setor choice
    if sel_setor:
        df_step = df_step[df_step["setor"].isin(sel_setor)]

    c7, c8 = st.columns([1, 1])
    with c7:
        opts_subsetor = unique_non_null(df_step["subsetor"])
        sel_subsetor = st.multiselect("CNAE Subsetor", options=opts_subsetor, default=[])
    if sel_subsetor:
        df_step = df_step[df_step["subsetor"].isin(sel_subsetor)]

    with c8:
        opts_segmento = unique_non_null(df_step["segmento"])
        sel_segmento = st.multiselect("CNAE Segmento", options=opts_segmento, default=[])
    if sel_segmento:
        df_step = df_step[df_step["segmento"].isin(sel_segmento)]

    st.markdown("### Results (both namings shown)")
    unified_results_table(df_step, widget_key="cnae")

# ====================== JP (PROPRIETARY) TAB ======================
with tab_prop:
    st.subheader("JP filters (proprietary)")

    # 1) Text search first so dropdown options reflect only matching companies
    df_base = common_text_filters(df, key_prefix="prop")

    # Progressive dependent options
    p1, p2, p3 = st.columns([1,1,1])

    # setor -> subsetor -> segmento (options restricted by previous picks)
    with p1:
        opts_prop_setor = unique_non_null(df_base["setor_progest"])
        sel_prop_setor = st.multiselect("Prop. Setor", options=opts_prop_setor, default=[])

    df_step = df_base
    if sel_prop_setor:
        df_step = df_step[df_step["setor_progest"].isin(sel_prop_setor)]

    with p2:
        opts_prop_subsetor = unique_non_null(df_step["subsetor_progest"])
        sel_prop_subsetor = st.multiselect("Prop. Subsetor", options=opts_prop_subsetor, default=[])

    if sel_prop_subsetor:
        df_step = df_step[df_step["subsetor_progest"].isin(sel_prop_subsetor)]

    with p3:
        opts_prop_segmento = unique_non_null(df_step["segmento_progest"])
        sel_prop_segmento = st.multiselect("Prop. Segmento", options=opts_prop_segmento, default=[])

    if sel_prop_segmento:
        df_step = df_step[df_step["segmento_progest"].isin(sel_prop_segmento)]

    st.markdown("### Results (both namings shown)")
    unified_results_table(df_step, widget_key="prop")

# ---------------- Notes ----------------
with st.expander("Notes"):
    st.markdown(
        """
- The dropdowns are **dependent**: once you filter a category, other fields only show non-null options available in the remaining data.
- **CNAE Primário (descrição)** and **CNAE Secundário (descrição)** are emphasized and applied first.
- **CNAE secundários** are treated as lists in `cnae_secundarios_norm`/`cnae_secundarios_descricao`; use the ANY/ALL switches to control matching behavior.
- All text searches are substring-based; dropdowns are exact-match multi-selects.
"""
    )
