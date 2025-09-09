# app.py
import re
from pathlib import Path
from typing import List, Optional, Set

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

TOP_RANKING_BASENAME = "top_123_companies_list_DECADE_and_LAST"  # we'll try .xlsx/.xls/.csv

# ---------------- Helpers ----------------
def format_cnpj(raw: str) -> str:
    """Format a 14-digit CNPJ as 00.000.000/0000-00."""
    if not isinstance(raw, str):
        return ""
    s = re.sub(r"\D", "", raw).zfill(14)[-14:]
    if len(s) != 14:
        return raw or ""
    return f"{s[0:2]}.{s[2:5]}.{s[5:8]}/{s[8:12]}-{s[12:14]}"

def normalize_cnpj_digits(raw: str) -> str:
    """Normalize to 14-digit only-digits string (left-padded) using the last 14 digits if longer."""
    s = re.sub(r"\D", "", str(raw or ""))
    if not s:
        return ""
    return s[-14:].zfill(14)

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
    df["cnpj_norm"] = df["cnpj"].map(normalize_cnpj_digits)
    df["__sec_code_list__"] = df["cnae_secundarios_norm"].map(split_list)
    df["__sec_desc_list__"] = df.apply(split_sec_descriptions, axis=1)

    return df

@st.cache_data(show_spinner=False)
def load_top_ranking_set() -> Set[str]:
    """
    Try to load the TOP ranking file from the working directory.
    Accepts .xlsx, .xls, .csv. Detects the first column whose name contains 'cnpj' (case-insensitive).
    Returns a set of normalized CNPJs (14-digit strings).
    """
    candidates = [
        Path(f"{TOP_RANKING_BASENAME}.xlsx"),
        Path(f"{TOP_RANKING_BASENAME}.xls"),
        Path(f"{TOP_RANKING_BASENAME}.csv"),
    ]
    path: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if path is None:
        return set()

    if path.suffix.lower() in [".xlsx", ".xls"]:
        tdf = pd.read_excel(path, dtype="string")
    else:
        tdf = pd.read_csv(path, dtype="string")

    # find CNPJ-like column
    cnpj_cols = [c for c in tdf.columns if "cnpj" in c.lower()]
    if not cnpj_cols:
        return set()

    col = cnpj_cols[0]
    return set(tdf[col].map(normalize_cnpj_digits).dropna().tolist())

def add_top_flag(df: pd.DataFrame, top_set: Set[str]) -> pd.DataFrame:
    df = df.copy()
    df["__IN_TOP__"] = df["cnpj_norm"].map(lambda x: x in top_set if isinstance(x, str) else False)
    return df

def company_search(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    st.subheader("🔎 Search")
    name_q = st.text_input("Company name contains", value="", key=f"{key_prefix}_name").strip()
    out = df
    if name_q:
        out = out[out["input_company"].fillna("").str.contains(name_q, case=False, na=False)]
    return out

def style_top_rows(view: pd.DataFrame):
    """Highlight rows where TOP RANKING == 'TOP' with light green background."""
    def _row_style(row: pd.Series):
        color = "background-color: #eaffea" if row.get("TOP RANKING", "") == "TOP" else ""
        return [color] * len(row)
    return view.style.apply(_row_style, axis=1)

def unified_results_table(df_filtered: pd.DataFrame, total_count: int, *, widget_key: str):
    """
    Build the final results view:
    - Add TOP RANKING and # columns at the left
    - Emphasize primary & secondary descriptions early
    - Remove 'cnpj' raw column
    - Show count "Empresas: X/Y"
    - Color TOP rows green
    """
    # Prepare label columns
    top_label = df_filtered["__IN_TOP__"].map(lambda b: "TOP" if bool(b) else "")
    idx_col = pd.Series(range(1, len(df_filtered) + 1), index=df_filtered.index, dtype="int")

    # Base view (order: TOP, #, Empresa, CNPJ_fmt, descriptions, rest)
    cols_order = [
        "input_company","CNPJ_fmt",
        "cnae_descricao",  # primary description (important)
        "cnae_secundarios_descricao",  # secondary description (important)
        "cnae_primario",
        "setor","subsetor","segmento",
        "cnae_secundarios_norm","cnae_secundarios_pairs",
        "setor_progest","subsetor_progest","segmento_progest",
        "cnae7","source","confidence",
    ]
    present = [c for c in cols_order if c in df_filtered.columns]
    view = df_filtered[present].rename(columns={
        "input_company": "Empresa",
        "CNPJ_fmt": "CNPJ (formatado)",
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

    # Insert left columns
    view.insert(0, "#", idx_col)
    view.insert(0, "TOP RANKING", top_label)

    # KPI count
    st.markdown(f"**Empresas: {len(view):,}/{total_count:,}**")

    # Render with style (highlight TOP)
    styled = style_top_rows(view)
    st.dataframe(styled, use_container_width=True, key=f"df_{widget_key}")

    # Download CSV (without styles)
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name=f"filtered_companies_{widget_key}.csv",
        mime="text/csv",
        key=f"download_{widget_key}",
    )

# ---------------- Data ----------------
df_raw = load_csv()
top_set = load_top_ranking_set()
df = add_top_flag(df_raw, top_set)

# ---------------- Tabs ----------------
tab_cnae, tab_prop = st.tabs(["🏷️ CNAE filters", "🏷️ JP filters (proprietary)"])

# ============================ CNAE TAB ============================
with tab_cnae:
    st.subheader("Filter by CNAE")

    # Search first so options reflect current subset
    df_base = company_search(df, key_prefix="cnae")

    # --- Emphasize descriptions first (most important) ---
    c1, c2 = st.columns([1.2, 1.2])
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
        # Secondary description options depend on primary selection
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

    # CNAE Setor (dependent on previous filters)
    opts_setor = unique_non_null(df_step["setor"])
    sel_setor = st.multiselect("CNAE Setor", options=opts_setor, default=[])

    if sel_setor:
        df_step = df_step[df_step["setor"].isin(sel_setor)]

    # Toggle: only TOP ranking companies
    only_top = st.checkbox("Apenas empresas no ranking TOP", value=False, key="only_top_cnae")
    if only_top:
        df_step = df_step[df_step["__IN_TOP__"]]

    st.markdown("### Results (both namings shown)")
    unified_results_table(df_step, total_count=len(df), widget_key="cnae")

# ====================== JP (PROPRIETARY) TAB ======================
with tab_prop:
    st.subheader("JP filters (proprietary)")

    # Search first so options reflect current subset
    df_base = company_search(df, key_prefix="prop")

    # Progressive dependent options (as before)
    p1, p2, p3 = st.columns([1,1,1])

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

    # Toggle: only TOP ranking companies
    only_top = st.checkbox("Apenas empresas no ranking TOP", value=False, key="only_top_prop")
    if only_top:
        df_step = df_step[df_step["__IN_TOP__"]]

    st.markdown("### Results (both namings shown)")
    unified_results_table(df_step, total_count=len(df), widget_key="prop")

# ---------------- Notes ----------------
with st.expander("Notes"):
    st.markdown(
        """
- The dropdowns are **dependent**: once you filter a category, other fields only show non-null options available in the remaining data.
- **CNAE Primário (descrição)** and **CNAE Secundário (descrição)** are emphasized and applied first.
- Rows that belong to the **TOP ranking list** are highlighted in light green; toggle “Apenas empresas no ranking TOP” to restrict results.
- All text searches are substring-based; dropdowns are exact-match multi-selects.
"""
    )
