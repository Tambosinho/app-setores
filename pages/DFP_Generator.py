# DFP_Generator.py
# Visualizador hierárquico de DFP (BP/DRE/DFC/DVA) a partir de um CSV "wide".
# - CSV com auto-detecção de delimitador (',' ou ';')
# - "Anos" via select_slider (discreto) para impedir anos fora da base
# - DRE com ordenação IFRS por nome de conta (robusto a variações)

from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# ================================
# 0) BASIC SETUP
# ================================
st.set_page_config(page_title="DFP Viewer", layout="wide")
st.title("📊 Visualizador de Demonstrações Financeiras (CVM)")

CSV_PATH = Path("financials_wide_with_industry.csv")
ID_COLS = ["empresa", "cnpj", "ano"]
ACCOUNT_COL_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*-\s*(.+?)\s*$")

# ================================
# 1) HELPERS
# ================================
def _doc_from_code_root(code_root: str) -> str:
    if code_root in {"1", "2"}: return "Balanço Patrimonial"
    if code_root == "3":        return "DRE"
    if code_root == "6":        return "DFC"
    if code_root == "7":        return "DVA"
    return "Outros"

def _indent(level: int, name: str) -> str:
    # 1 tab = 4 espaços (NBSP para não colapsar)
    return ("\u00A0" * 4 * level) + name

def _format_brl(x) -> str:
    if pd.isna(x): return ""
    try:
        return ("R$ " + f"{float(x):,.2f}").replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def _code_to_sortkey(code: object) -> tuple:
    if not isinstance(code, str): return (9999,)
    parts = []
    for p in str(code).split("."):
        try: parts.append(int(p))
        except Exception: parts.append(0)
    return tuple(parts) if parts else (9999,)

def _norm(s: str) -> str:
    """Normaliza nome: sem acentos, maiúsculas, espaços únicos, remove pontuação supérflua."""
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = s.upper()
    s = s.replace("&", "E").replace("E / OU", "E/OU").replace("E /OU", "E/OU").replace("E/ OU", "E/OU")
    s = re.sub(r"[^A-Z0-9/ ()]", " ", s)          # mantém letras/números, / e parênteses
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------- Ordem IFRS para DRE (regex sobre nome normalizado), de cima para baixo --------
_DRE_ORDER_PATTERNS = [
    # Receita / Custo / Bruto
    r"^RECEITA DE VENDA DE BENS( E/OU)? SERVICOS$",                         # RECEITA DE VENDA DE BENS E/OU SERVICOS
    r"^CUSTO DOS BENS( E/OU)? SERVICOS VENDIDOS$",                          # CUSTO DOS BENS E/OU SERVICOS VENDIDOS
    r"^RESULTADO BRUTO$",

    # Despesas/Receitas operacionais
    r"(DESPESAS/? ?RECEITAS OPERACIONAIS|OUTRAS DESPESAS E RECEITAS OPERACIONAIS)",
    r"^DESPESAS? COM VENDAS$",
    r"^DESPESAS? ADMINISTRATIVAS$",
    r"(PERDAS? PELA NAO RECUPERABILIDADE DE ATIVOS|PERDAS? POR REDUCAO AO VALOR RECUPERAVEL)",
    r"^OUTRAS RECEITAS OPERACIONAIS$",
    r"^OUTRAS DESPESAS OPERACIONAIS$",      # (header)
    r"^OUTRAS DESPESAS OPERACIONAIS$",      # (sublinha repetida em alguns layouts)
    r"^DESPESAS? TRIBUTARIAS$",
    r"^OUTROS$",
    r"^RESULTADO DE EQUIVALENCIA PATRIMONIAL$",

    # EBIT
    r"^RESULTADO ANTES DO RESULTADO FINANCEIRO E DOS TRIBUTOS$",

    # Financeiro (IFRS apresenta geralmente líquido, mas mantemos blocos)
    r"^RESULTADO FINANCEIRO$",
    r"^RECEITAS? FINANCEIRAS$",
    r"^RECEITAS? DE JUROS( E RENDIMENTOS( SIMILARES)?)?$",
    r"GANH(OS)?/? ?\(?PERDAS?\)? .*ATIV.*PASSIV.*FINANCEI",
    r"^RECEITAS? DE DIVIDENDOS$",
    r"^RECEITAS? DE SERVICOS? FINANCEIROS$",
    r"^RECEITAS? DE PREMIOS? DE OPER\.? SEGUROS? E PREVIDENCIA$",
    r"^RECEITAS? DE PREMIOS? DE OPER\.? SEGUROS? E PREVIDENCIA E CAPITALIZACAO$",
    r"(OPERACOES? DE CAMBIO|VARIACAO CAMBIAL.*TRANSACOES EXTERIOR)",

    r"^DESPESAS? FINANCEIRAS$",
    r"^DESPESAS? TRIBUTARIAS$",       # (algumas cias classificam dentro do financeiro)
    r"^DESPESAS? DE JUROS( E RENDIMENTOS)?$",
    r"(PERDAS? COM CREDITO|RECEBIVEIS|SINISTROS)",
    r"PROVISO(ES)? DE SEGUROS? E PREVIDENCIA PRIVADA",

    # EBT / Impostos
    r"^RESULTADO ANTES DOS TRIBUTOS SOBRE O LUCRO$",
    r"IMPOSTO DE RENDA E CONTRIBUICAO SOCIAL .* LUCRO",
    r"^CORRENT(E)?$",
    r"^DIFERID[OA]S?$",

    # Resultado líquido / EPS
    r"^RESULTADO LIQUIDO DAS OPERACOES CONTINUADAS$",
    r"^RESULTADO LIQUIDO DAS OPERACOES DESCONTINUADAS$",
    r"LUCRO/?PREJUIZO .* OPERACOES DESCONTINUADAS",
    r"GANH(OS)?/? ?\(?PERDAS?\)? .* ATIVOS .* OPERACOES DESCONTINUADAS",
    r"^LUCRO OU PREJUIZO LIQUIDO CONSOLIDADO DO PERIODO$",
    r"ATRIBUIDO A SOCIOS? DA EMPRESA CONTROLADORA",
    r"ATRIBUIDO A SOCIOS? NAO CONTROLADORES",
    r"^LUCRO POR ACAO ?\(?R\$/?ACAO\)?$",
    r"^LUCRO BASICO POR ACAO$",
    r"^ON$",
    r"^PN$",
    r"^LUCRO DILUIDO POR ACAO$",
    r"^ON$",
    r"^PN$",
]
_DRE_ORDER_REGEX = [re.compile(p) for p in _DRE_ORDER_PATTERNS]

def _dre_priority(name: str) -> int:
    n = _norm(name)
    for i, rgx in enumerate(_DRE_ORDER_REGEX):
        if rgx.search(n):
            return i
    return 10_000  # não mapeado → vai para o final relativo (ordem por código)

# --- CSV readers (auto-delimiter: ',' ou ';') ---
CSV_KW = dict(sep=None, engine="python")

@st.cache_data(show_spinner=False)
def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, **CSV_KW).columns)

@st.cache_data(show_spinner=False)
def read_meta_min(path: Path) -> pd.DataFrame:
    hdr = read_header(path)
    use = [c for c in ID_COLS if c in hdr]
    if not use: return pd.DataFrame()
    df = pd.read_csv(path, usecols=use, dtype={"empresa": "string", "cnpj": "string"}, **CSV_KW)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def build_accounts_index(path: Path) -> pd.DataFrame:
    cols = read_header(path)
    rows = []
    for col in cols:
        m = ACCOUNT_COL_PATTERN.match(col)
        if not m: continue
        code, name = m.group(1), m.group(2).strip()
        top = code.split(".")[0]
        rows.append({"full": col, "code": code, "name": name,
                     "doc": _doc_from_code_root(top), "level": code.count(".")})
    idx = pd.DataFrame(rows)
    return idx[idx["doc"].isin(["Balanço Patrimonial", "DRE", "DFC", "DVA"])].copy()

@st.cache_data(show_spinner=True)
def load_company_slice(path: Path, cnpj: str, years: list[int], account_cols: list[str]) -> pd.DataFrame:
    hdr = read_header(path)
    needed_cols = [c for c in ID_COLS if c in hdr] + [c for c in account_cols if c in hdr]
    if not needed_cols: return pd.DataFrame()

    df = pd.read_csv(path, usecols=needed_cols, dtype={"empresa": "string", "cnpj": "string"}, **CSV_KW)
    df = df[df["cnpj"].astype(str) == str(cnpj)]
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        if years: df = df[df["ano"].isin(years)]
        df = df.sort_values("ano")

    for c in [c for c in account_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_year_pivot(df_slice: pd.DataFrame, account_cols: list[str], accounts_index: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_slice.empty or not account_cols: return pd.DataFrame(), pd.DataFrame()
    available = [c for c in account_cols if c in df_slice.columns]
    if not available: return pd.DataFrame(), pd.DataFrame()

    mat = df_slice.set_index("ano")[available].T
    # Só mantém colunas existentes para os anos selecionados (discretos)
    mat = mat[[y for y in years if y in mat.columns]]

    info = accounts_index.set_index("full").loc[available].copy()
    keep = ~mat.isna().all(axis=1)
    mat, info = mat[keep], info[keep]
    if info.empty: return pd.DataFrame(), pd.DataFrame()

    # ===== ORDENAR =====
    # DRE: prioridade por nome IFRS, depois código
    if (not info.empty) and (info["doc"].iloc[0] == "DRE"):
        info["_prio"] = info["name"].apply(_dre_priority)
    else:
        info["_prio"] = 10_000

    info["_codesort"] = info["code"].apply(_code_to_sortkey)
    info = info.sort_values(["_prio", "_codesort"])
    mat = mat.loc[info.index]
    info = info.drop(columns=["_prio", "_codesort"])

    # ===== MONTAR TABELAS =====
    numeric_df = pd.DataFrame({
        "Código": info["code"].tolist(),
        "Conta":  [_indent(l, n) for l, n in zip(info["level"], info["name"])],
    }).reset_index(drop=True)
    numeric_df = pd.concat([numeric_df, mat.reset_index(drop=True)], axis=1)

    display_df = numeric_df.copy()
    for y in years:
        if y in display_df.columns:
            display_df[y] = display_df[y].apply(_format_brl)
    return display_df, numeric_df

# ================================
# 2) FILE CHECKS
# ================================
if not CSV_PATH.exists():
    st.error(f"Arquivo não encontrado: {CSV_PATH.resolve()}")
    st.stop()

# ================================
# 3) LOAD META + ACCOUNTS INDEX
# ================================
try:
    meta = read_meta_min(CSV_PATH)
    acc_idx = build_accounts_index(CSV_PATH)
except pd.errors.EmptyDataError:
    st.error("Não foi possível ler o arquivo. Verifique se o CSV não está vazio e se o caminho está correto.")
    st.stop()

if meta.empty:
    st.warning("Não foi possível identificar colunas de ID (empresa/cnpj/ano).")
    st.stop()
if acc_idx.empty:
    st.warning("Não foram encontradas colunas de contas no formato '{código} - {nome}'.")
    st.stop()

# ================================
# 4) UI CONTROLS
# ================================
meta_company = (
    meta.dropna(subset=["empresa", "cnpj"])
        .drop_duplicates(subset=["empresa", "cnpj"])
        .assign(label=lambda d: d["empresa"].str.strip() + " — " + d["cnpj"].str.strip())
        .sort_values("label")
)
company_label = st.selectbox("Empresa (CNPJ)", options=meta_company["label"].tolist(),
                             index=0 if len(meta_company) else None,
                             placeholder="Selecione a empresa...")

sel_row = meta_company.loc[meta_company["label"] == company_label].iloc[0]
sel_cnpj = sel_row["cnpj"]

company_years = (
    meta.loc[meta["cnpj"] == sel_cnpj, "ano"]
        .dropna().astype(int).sort_values().unique().tolist()
)

# select_slider com valores discretos (somente anos existentes)
if not company_years:
    st.info("Não há anos disponíveis para a empresa selecionada.")
    st.stop()

if len(company_years) == 1:
    year_choice = st.select_slider("Ano", options=company_years, value=company_years[0])
    sel_years = [int(year_choice)]
else:
    yr_start, yr_end = company_years[0], company_years[-1]
    year_choice = st.select_slider(
        "Ano(s) (faixa discreta; para 1 ano, selecione o mesmo valor nas duas alças)",
        options=company_years, value=(yr_start, yr_end),
    )
    if isinstance(year_choice, tuple):
        i0 = company_years.index(int(year_choice[0]))
        i1 = company_years.index(int(year_choice[1]))
        if i0 > i1: i0, i1 = i1, i0
        sel_years = company_years[i0:i1+1]
    else:
        sel_years = [int(year_choice)]

doc_options = ["Balanço Patrimonial", "DRE", "DFC", "DVA"]
sel_doc = st.selectbox("Documento", options=doc_options, index=1)  # DRE como default se preferir

doc_accounts = acc_idx[acc_idx["doc"] == sel_doc].sort_values("code")
account_cols_for_doc = doc_accounts["full"].tolist()
if not account_cols_for_doc:
    st.info(f"Não foram encontradas contas para **{sel_doc}**.")
    st.stop()

# ================================
# 5) LOAD SLICE + RENDER TABLE
# ================================
with st.spinner("Carregando dados selecionados..."):
    df_slice = load_company_slice(CSV_PATH, sel_cnpj, sel_years, account_cols_for_doc)

if df_slice.empty:
    st.info("Não há dados para a combinação selecionada (empresa/anos).")
    st.stop()

display_df, numeric_df = make_year_pivot(df_slice, account_cols_for_doc, acc_idx[acc_idx["doc"] == sel_doc], sel_years)

st.subheader(f"{sel_doc} — {company_label}")
st.caption("Valores em Reais (R$). Linhas sem valores nos anos selecionados são ocultadas automaticamente.")

if display_df.empty:
    st.info("Nenhuma conta com valores para os anos selecionados.")
else:
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="⬇️ Baixar tabela (CSV, valores numéricos)",
        data=numeric_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{sel_doc.replace(' ', '_').lower()}_{sel_cnpj}.csv",
        mime="text/csv",
    )

with st.expander("Notas e considerações técnicas"):
    st.markdown(
        """
- **DRE (IFRS)**: ordem customizada por **nome da conta** (regex em nome normalizado) para priorizar:
  Receita → Custo → Resultado Bruto → (Despesas/Receitas operacionais) → EBIT →
  Resultado Financeiro → EBT → Imposto (Corrente, Diferido) → Resultado Líquido
  (Continuadas/Descontinuadas) → Atribuição (Controladora/NCI) → EPS (Básico/Diluído, ON/PN).
- **Anos**: `select_slider` com opções **apenas dos anos existentes** para a empresa.
- **Indentação**: 1 tab (4 NBSP) por nível do código.
- **Desempenho**: leitura seletiva — somente IDs + contas do documento escolhido.
"""
    )
