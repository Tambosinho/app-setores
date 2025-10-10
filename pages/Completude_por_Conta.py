# pages/Completude_por_Conta.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set, Dict

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Completude por Conta", layout="wide")
st.markdown("<style>.block-container{max-width:96vw}</style>", unsafe_allow_html=True)
st.title("Completude por Conta (hierarquia, sem agregação)")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
DRE_ORDER = [
    "3.14", "3.20", "3.15", "3.21", "3.16",
    "3.17", "3.22", "3.04", "3.23", "3.18",
    "3.19", "3.05", "3.06", "3.07", "3.08",
    "3.09", "3.10",
]
DRE_RANK = {code: i for i, code in enumerate(DRE_ORDER)}

def coerce_percent_strict(series: pd.Series) -> pd.Series:
    def _parse(x):
        if pd.isna(x):
            return None
        s = str(x).strip().replace("\xa0"," ").replace("\u200b"," ")
        s = s.replace("%","")
        s = re.sub(r"[^0-9,.\-]", "", s)
        if "," in s and "." in s:
            s = s.replace(".","").replace(",",".")
        elif "," in s:
            s = s.replace(",",".")
        try:
            v = float(s)
        except Exception:
            return None
        return max(0.0, min(100.0, v))
    return series.apply(_parse)

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\xa0", " ").replace("\u200b", " ")
    return re.sub(r"\s+", " ", s).strip()

def find_col(df: pd.DataFrame, *patterns) -> str:
    pats = [p.lower() for p in patterns]
    for c in df.columns:
        if any(p in str(c).lower() for p in pats):
            return c
    raise KeyError(f"Coluna não encontrada para padrões: {patterns}")

def lighten_hex(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount); g = int(g + (255 - g) * amount); b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

def extract_root_digit(code: str, label: str) -> Optional[str]:
    m = re.match(r"\s*(\d+)", str(code or ""))
    if m: return m.group(1)[0]
    m = re.match(r"\s*(\d+)", str(label or ""))
    if m: return m.group(1)[0]
    return None

# ------------------------------------------------------------
# Core renderer
# ------------------------------------------------------------
def render_completude_table(
    df_in: pd.DataFrame,
    *,
    title: Optional[str] = None,
    height: int = 700,
    width: int = 1600,
    base_palette: Optional[Dict[str, str]] = None,
    allowed_roots: Optional[Set[str]] = None,
):
    if base_palette is None:
        base_palette = {
            "1": "#ff7a6b",  # vermelho
            "2": "#65c2ff",  # azul
            "3": "#c489ff",  # lilás
            "4": "#ffac64",  # laranja
            "5": "#5fffdf",  # teal
            "6": "#ffe16b",  # amarelo
            "7": "#df8fff",  # roxo
        }
    fallback_base = "#95a5a6"  # cinza

    df = df_in.copy()
    df.columns = [norm_text(c) for c in df.columns]

    col_conta   = find_col(df, "conta (hierarquia)", "conta", "hierarquia")
    col_codigo  = find_col(df, "código", "codigo")
    col_comp    = find_col(df, "completude")
    df["_comp_pct"] = coerce_percent_strict(df[col_comp])

    col_emp_min = find_col(df, "empresas", "minorit")
    lvl1 = find_col(df, "level 1")
    lvl2 = find_col(df, "level 2")
    lvl3 = find_col(df, "level 3")
    lvl4 = find_col(df, "level 4")
    level_cols = [lvl1, lvl2, lvl3, lvl4]

    for c in level_cols:
        df[c] = (
            df[c].astype(str)
                 .str.replace("\xa0"," ",regex=False)
                 .str.replace("\u200b"," ",regex=False)
                 .str.strip()
                 .replace({"nan":"", "None":""})
        )

    def build_levels_path(row):
        segs = [row[c] for c in level_cols if str(row[c]).strip()]
        if segs:
            key = "||".join(segs)
            parent = "||".join(segs[:-1]) if len(segs) > 1 else ""
            level = len(segs)
            label = segs[-1]
            return key, parent, level, label
        code = (
            str(row[col_codigo]).strip()
            .replace("\xa0"," ").replace(" ","")
            .replace(",", ".")
        )
        key = code
        parent = code.rsplit(".", 1)[0] if "." in code else ""
        level = code.count(".") + 1 if code else 1
        label = row[col_conta]
        return key, parent, level, label

    keys, parents, levels, labels = [], [], [], []
    for _, r in df.iterrows():
        k, p, lv, lb = build_levels_path(r)
        keys.append(k); parents.append(p); levels.append(lv); labels.append(lb)

    df["key"] = keys; df["parent"] = parents; df["level"] = levels; df["label"] = labels

    key_set = set(df["key"])
    l1_series = (
        df[lvl1].astype(str)
                .str.replace("\xa0", " ", regex=False)
                .str.replace("\u200b", " ", regex=False)
                .str.strip()
                .replace({"nan": "", "None": ""})
    )
    l1_unique = [v for v in l1_series.unique() if v]
    missing_l1 = [v for v in l1_unique if v not in key_set]

    if missing_l1:
        synth = pd.DataFrame({
            "label":   missing_l1,
            "key":     missing_l1,
            "parent":  ["" for _ in missing_l1],
            "level":   [1 for _ in missing_l1],
            "is_root": [True for _ in missing_l1],
            col_codigo:  ["" for _ in missing_l1],
            col_comp:    [None for _ in missing_l1],
            col_emp_min: ["" for _ in missing_l1],
        })
        cols_keep = [col_conta, col_codigo, col_comp, col_emp_min, "label", "key", "parent", "level", "is_root"]
        synth[col_conta] = synth["label"]
        df = pd.concat([df, synth[cols_keep]], ignore_index=True)

    key_set = set(df["key"])
    df["is_root"] = ~df["parent"].isin(key_set)
    parent_ids = set(df.loc[df["parent"] != "", "parent"])

    df["root_digit"] = [extract_root_digit(r[col_codigo], r["label"]) for _, r in df.iterrows()]
    if allowed_roots:
        df = df[df["root_digit"].isin(allowed_roots)].copy()
        key_set = set(df["key"])
        df["is_root"] = ~df["parent"].isin(key_set)
        parent_ids = set(df.loc[df["parent"] != "", "parent"])

    def bg_for_row(root_digit: Optional[str], level: int) -> str:
        base = base_palette.get(root_digit or "", fallback_base)
        amount = min(0.20 + 0.15 * max(level - 1, 0), 0.85)
        return lighten_hex(base, amount)

    rows = []
    for _, r in df.sort_values(["key"]).iterrows():
        lvl = int(r["level"]) if pd.notna(r["level"]) else 1
        indent_px = max(lvl - 1, 0) * 18
        has_children = r["key"] in parent_ids
        caret_html = '<span class="toggle">▸</span>' if has_children else '<span class="spacer"></span>'

        is_empty = pd.isna(r["_comp_pct"])
        if is_empty:
            comp_html = '''
              <div class="pb-wrap pb-empty">
                <div class="pb"><div class="fill" style="width:0%"></div></div>
                <div class="pct"></div>
              </div>
            '''
        else:
            pct = float(r["_comp_pct"])
            comp_html = f'''
              <div class="pb-wrap">
                <div class="pb"><div class="fill" style="width:{pct:.2f}%"></div></div>
                <div class="pct">{pct:.2f}%</div>
              </div>
            '''

        bg = bg_for_row(r.get("root_digit"), lvl)

        rows.append(f"""
          <tr class="node" data-id="{r['key']}" data-parent="{r['parent']}" data-level="{lvl}" data-root="{1 if r['is_root'] else 0}" style="background-color:{bg}">
            <td>{caret_html}<span class="label" style="padding-left:{indent_px}px">{r['label']}</span></td>
            <td>{r[col_codigo]}</td>
            <td class="comp">{comp_html}</td>
            <td>{r[col_emp_min]}</td>
          </tr>
        """)

    rows_html = "".join(rows)

    html = f"""
    <style>
      table {{ width:100%; border-collapse:collapse; font-family:Inter, system-ui; }}
      thead th {{ text-align:left; font-weight:600; border-bottom:1px solid #e6e6e6; padding:10px 8px; }}
      td {{ padding:8px; border-bottom:1px solid #f2f2f2; white-space:nowrap; background: inherit; }}
      td.num {{ text-align:right; width:120px; }}
      tr.hidden {{ display:none; }}
      .toggle {{ margin-right:6px; cursor:pointer; user-select:none; display:inline-block; width:14px; }}
      .spacer {{ display:inline-block; width:14px; margin-right:6px; }}
      tbody tr:hover td {{ filter: brightness(0.98); }}
      td.comp {{ width: 260px; }}
      .comp .pb-wrap {{ display:flex; align-items:center; gap:10px; }}
      .comp .pb      {{ width: 200px; height: 10px; background:#eee; border-radius:999px; overflow:hidden; }}
      .comp .pb > .fill {{
        height: 100%;
        background: #2ecc71;              /* the green fill */
        transition: width .2s ease-in-out; /* nice animation (optional) */
        }}

      .comp .pct     {{ min-width: 58px; text-align:right; font-variant-numeric: tabular-nums; color:#333; }}
      .section-title {{ font-weight:600; margin: 12px 0 6px 0; }}
      .comp .pb-wrap.pb-empty {{ opacity: 0.65; }}
      .comp .pb-wrap.pb-empty .pb {{ background: transparent; }}
      .comp .pb-wrap.pb-empty .fill {{ display: none; }}
      .comp .pb-wrap.pb-empty .pct {{ visibility: hidden; }}
    </style>

    {"<div class='section-title'>" + title + "</div>" if title else ""}
    <div style="overflow-x:auto;">
    <table id="treetable-{id(df)}" style="min-width: 1200px;">
      <thead>
        <tr>
          <th>Nome da conta</th>
          <th>Código</th>
          <th>Completude</th>
          <th>Empresas minoritárias</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    </div>

    <script>
    (function() {{
      const tblId = "treetable-{id(df)}";
      const rows = Array.from(document.querySelectorAll('#' + tblId + ' tbody tr'));

      // mostra só nós raiz inicialmente
      rows.forEach(tr => {{
        if (tr.dataset.root !== "1") tr.classList.add('hidden');
      }});

      rows.forEach(tr => {{
        const tg = tr.querySelector('.toggle');
        if (!tg) return;
        tg.addEventListener('click', (e) => {{
          e.stopPropagation();
          const id = tr.dataset.id;
          const isOpen = tr.classList.toggle('open');
          tg.textContent = isOpen ? '▾' : '▸';
          toggleChildren(id, isOpen);
        }});
      }});

      function toggleChildren(parentId, show) {{
        rows.forEach(tr => {{
          if (tr.dataset.parent === parentId) {{
            if (show) {{
              tr.classList.remove('hidden');
              if (!tr.classList.contains('open')) hideDescendants(tr.dataset.id);
            }} else {{
              tr.classList.add('hidden');
              hideDescendants(tr.dataset.id);
              const tg = tr.querySelector('.toggle'); if (tg) tg.textContent = '▸';
              tr.classList.remove('open');
            }}
          }}
        }});
      }}

      function hideDescendants(id) {{
        rows.forEach(tr => {{
          const parent = tr.dataset.parent || "";
          if (parent === id ||
              parent.startsWith(id + (id.includes('||') ? '||' : '.'))) {{
            tr.classList.add('hidden');
          }}
        }});
      }}
    }})();
    </script>
    """
    st.components.v1.html(html, height=height, width=width, scrolling=True)

# ------------------------------------------------------------
# Data loading (arquivos no repositório)
# ------------------------------------------------------------
CANDIDATE_DIRS = [Path("."), Path("data"), Path("datasets"), Path("assets"), Path("files")]

FILES = {
    "NFIN_400": "completude_contas_formatado_400_NAO_FINANCEIRAS_adaptado_powerbi.xlsx",
    "FIN_400":  "completude_contas_formatado_400_FINANCEIRAS_adaptado_powerbi.xlsx",
    "NFIN_120": "completude_contas_formatado_120_NAO_FINANCEIRAS_adaptado_powerbi.xlsx",
    "FIN_120":  "completude_contas_formatado_120_FINANCEIRAS_adaptado_powerbi.xlsx",
}

def resolve_path(fname: str) -> Optional[Path]:
    for base in CANDIDATE_DIRS:
        p = base / fname
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def load_required(label_key: str) -> Optional[pd.DataFrame]:
    fname = FILES[label_key]
    p = resolve_path(fname)
    if p is None:
        st.error(f"Arquivo não encontrado: `{fname}` (procurei em {', '.join(str(d) for d in CANDIDATE_DIRS)})")
        return None
    try:
        return load_excel(p)
    except Exception as e:
        st.error(f"Falha ao ler `{p}`: {e}")
        return None

# ------------------------------------------------------------
# Filtros principais (no topo, horizontais)
# ------------------------------------------------------------
DOC_LABEL_TO_ROOT = {
    "TODOS": None,  # mostra 1,2,3,6,7
    "Balanço Patrimonial Ativo": "1",
    "Balanço Patrimonial Passivo": "2",
    "Demonstrações do Resultado do Exercício": "3",
    "Demonstrações do Fluxo de Caixa": "6",
    "Demonstrações de Valor Adicionado": "7",
}
doc_labels = list(DOC_LABEL_TO_ROOT.keys())  # "TODOS" fica em primeiro

col1, col2, col3 = st.columns(3)
with col1:
    tipo_empresa = st.selectbox(
        "Tipo de empresa",
        options=["Não Financeiras", "Financeiras"],
        index=0,
        key="tipo_empresa_select",
    )
with col2:
    conjunto = st.selectbox(
        "Conjunto",
        options=["Top 120", "Todas as empresas"],
        index=1,
        key="conjunto_select",
    )
with col3:
    doc_label = st.selectbox(
        "Documento",
        options=doc_labels,
        index=0,  # "TODOS" como padrão
        key="documento_select",
    )

# roots permitidas (TODOS => 1,2,3,6,7)
selected_root = DOC_LABEL_TO_ROOT[doc_label]
allowed_set: Set[str] = set("12367") if selected_root is None else {selected_root}

# ------------------------------------------------------------
# Carregamento dos 4 dataframes
# ------------------------------------------------------------
df_nfin_400 = load_required("NFIN_400")
df_fin_400  = load_required("FIN_400")
df_nfin_120 = load_required("NFIN_120")
df_fin_120  = load_required("FIN_120")

# Escolha do DF conforme filtros
if tipo_empresa == "Não Financeiras":
    df_all, df_top = df_nfin_400, df_nfin_120
else:
    df_all, df_top = df_fin_400, df_fin_120

df_chosen = df_top if conjunto == "Top 120" else df_all

# ------------------------------------------------------------
# Render
# ------------------------------------------------------------
titulo = f"{tipo_empresa} — {conjunto} — {doc_label}"
st.subheader(titulo)

if df_chosen is None:
    st.stop()

render_completude_table(
    df_chosen,
    title="",
    allowed_roots=allowed_set,
    height=700,
    width=1600,
)
