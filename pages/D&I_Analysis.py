st.set_page_config(
    page_title="FRE • Empregados",
    page_icon="👔",
    layout="wide"
)
alt.data_transformers.disable_max_rows()

# Small css polish
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
.block-container { padding-top: 1rem; }
.header-box { background: #0f172a; color: white; padding: 16px 18px; border-radius: 14px; }
.badge { background: #e2e8f0; color: #0f172a; padding: 2px 8px; border-radius: 999px; font-size: 0.9rem; }
.small { opacity: 0.8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# File map (exact filenames)
# -----------------------------
FILE_MAP = {
    ("Posição", "Gênero"): "fre_cia_aberta_empregado_posicao_declaracao_genero_2024.csv",
    ("Posição", "Raça"):   "fre_cia_aberta_empregado_posicao_declaracao_raca_2024.csv",
    ("Posição", "Idade"):  "fre_cia_aberta_empregado_posicao_faixa_etaria_2024.csv",
    ("Região",  "Gênero"): "fre_cia_aberta_empregado_local_declaracao_genero_2024.csv",
    ("Região",  "Raça"):   "fre_cia_aberta_empregado_local_declaracao_raca_2024.csv",
    ("Região",  "Idade"):  "fre_cia_aberta_empregado_local_faixa_etaria_2024.csv",
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Keeps CNPJ as string (leading zeros preserved)
    - Tries common encodings (utf-8, latin-1)
    - Normalizes basic column names found in FRE
    """
    encodings = ["utf-8", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                dtype={
                    "CNPJ_Companhia": "string",
                    "CNPJ_Companhia ": "string",
                    "CNPJ_Cia": "string",
                    "CNPJ_CIA": "string",
                    "CNPJ": "string",
                },
                engine="python"
            )
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err

    # Normalize a few expected columns
    rename_map = {
        "CNPJ_Cia": "CNPJ_Companhia",
        "CNPJ_CIA": "CNPJ_Companhia",
        "CNPJ": "CNPJ_Companhia",
        "Nome_Companhia ": "Nome_Companhia",
        "Posicao": "Posicao",
        "Posição": "Posicao",   # safety
        "Local": "Local",
        "Regiao": "Local",
        "Região": "Local",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Clean CNPJ rendering (preserve original text with slashes/hyphens if present)
    if "CNPJ_Companhia" in df.columns:
        df["CNPJ_Companhia"] = df["CNPJ_Companhia"].astype("string")

    # Strip whitespace from all column names
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_all(base_dir: Path) -> dict:
    """Return {(Modo, Métrica): DataFrame} for the six inputs present."""
    all_dfs = {}
    for key, fname in FILE_MAP.items():
        fpath = base_dir / fname
        if fpath.exists():
            all_dfs[key] = load_csv(fpath)
    return all_dfs

def find_measure_columns(df: pd.DataFrame) -> list[str]:
    """
    The FRE tables shown have columns like:
    Quantidade_Ate30Anos, Quantidade_30a50Anos, Quantidade_Acima50Anos
    Quantidade_Branco, Quantidade_Preto, ...
    We'll treat any column starting with 'Quantidade' as a metric column.
    """
    return [c for c in df.columns if c.startswith("Quantidade")]

def melt_company(df: pd.DataFrame, entity_cols: list[str]) -> pd.DataFrame:
    """Melt wide 'Quantidade_*' columns to long 'Categoria/Valor'."""
    measures = find_measure_columns(df)
    if not measures:
        return pd.DataFrame()
    return df.melt(
        id_vars=entity_cols,
        value_vars=measures,
        var_name="Categoria",
        value_name="Valor"
    )

def tidy_label(label: str) -> str:
    """Prettify category labels (remove 'Quantidade_' and snake-ish bits)."""
    label = re.sub(r"^Quantidade_+", "", label)
    label = label.replace("_", " ")
    label = label.replace("a", " a ") if "a" in label and "Anos" in label else label
    label = re.sub(r"\s+", " ", label).strip()
    return label

def get_company_index(all_dfs: dict) -> pd.DataFrame:
    """
    Build a company list from whatever tables are available.
    Prefer Nome_Companhia + CNPJ_Companhia; drop duplicates.
    """
    frames = []
    for df in all_dfs.values():
        picks = []
        for col in ["CNPJ_Companhia", "Nome_Companhia"]:
            if col not in df.columns:
                # Skip tables that somehow don't have the basic ids
                break
        else:
            picks = df[["CNPJ_Companhia", "Nome_Companhia"]].copy()
            frames.append(picks)

    if not frames:
        return pd.DataFrame(columns=["CNPJ_Companhia", "Nome_Companhia"])

    idx = pd.concat(frames, ignore_index=True).dropna(subset=["Nome_Companhia"]).drop_duplicates()
    idx = idx.sort_values("Nome_Companhia", kind="stable").reset_index(drop=True)
    return idx

def company_rows(df: pd.DataFrame, cnpj: str, nome: str) -> pd.DataFrame:
    """
    Filter by (CNPJ or Name). Many FRE rows repeat the same company across subgroups.
    We'll keep all rows that match either to remain robust to tiny formatting differences.
    """
    mask = False
    if "CNPJ_Companhia" in df.columns and pd.notna(cnpj):
        mask = (df["CNPJ_Companhia"].astype(str) == str(cnpj))
    if "Nome_Companhia" in df.columns and pd.notna(nome):
        mask = mask | (df["Nome_Companhia"].astype(str) == str(nome))
    sub = df[mask].copy()
    return sub

def bar_chart(df_long: pd.DataFrame, stack_by: str | None) -> alt.Chart:
    """
    df_long must have columns: Categoria, Valor and optionally stack_by (e.g., Posicao or Local).
    """
    df_long = df_long.copy()
    df_long["Categoria"] = df_long["Categoria"].map(tidy_label)

    if stack_by and stack_by in df_long.columns:
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X("Categoria:N", sort="-y", title="Categoria"),
            y=alt.Y("sum(Valor):Q", title="Quantidade"),
            color=alt.Color(f"{stack_by}:N", title=stack_by),
            tooltip=["Categoria:N", f"{stack_by}:N", alt.Tooltip("sum(Valor):Q", title="Qtd")]
        )
    else:
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X("Categoria:N", sort="-y", title="Categoria"),
            y=alt.Y("Valor:Q", title="Quantidade"),
            tooltip=["Categoria:N", alt.Tooltip("Valor:Q", title="Qtd")]
        )
    return chart.properties(height=380)

# -----------------------------
# Data loading
# -----------------------------
with st.sidebar:
    st.header("⚙️ Arquivos")
    base_dir = st.text_input(
        "Pasta dos CSVs",
        value=str(Path.cwd()),
        help="Use a pasta que contém os 6 arquivos CSV listados no enunciado."
    )
    base_dir = Path(base_dir).expanduser().resolve()
    all_dfs = load_all(base_dir)
    missing = [f for f in FILE_MAP.values() if (base_dir / f).exists() is False]
    if missing:
        st.info("Arquivos faltando:\n- " + "\n- ".join(missing))

# Build company index
companies = get_company_index(all_dfs)
if companies.empty:
    st.stop()

# -----------------------------
# Step 1 — Company browser
# -----------------------------
st.title("📊 FRE – Empregados por Companhia")

with st.container():
    st.subheader("1) Escolha a companhia")
    col1, col2 = st.columns([2, 3])
    with col1:
        search = st.text_input("Buscar por nome ou CNPJ", "")
    view = companies.copy()
    if search.strip():
        s = search.strip().lower()
        view = view[view.apply(
            lambda r: s in str(r["Nome_Companhia"]).lower() or s in str(r["CNPJ_Companhia"]).lower(),
            axis=1
        )]
    # Dropdown for precise selection
    options = view.assign(_opt=view["Nome_Companhia"] + "  ·  " + view["CNPJ_Companhia"].fillna(""))._opt.tolist()
    pick = st.selectbox("Selecione a empresa", options, index=0)
    # Extract chosen row
    chosen = view.iloc[options.index(pick)] if options else view.iloc[0]
    cnpj_sel = chosen["CNPJ_Companhia"]
    nome_sel = chosen["Nome_Companhia"]

# -----------------------------
# Step 2 — Company page + Tabs
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("2) Página da Companhia")

# Header card
with st.container():
    st.markdown(
        f"""
        <div class="header-box">
            <div style="display:flex; gap:18px; align-items:center;">
                <div style="width:56px; height:56px; border-radius:14px; background:#334155; display:flex; align-items:center; justify-content:center; font-weight:700;">
                    {(nome_sel[:2] if isinstance(nome_sel, str) else '??').upper()}
                </div>
                <div>
                    <div style="font-size:1.35rem; font-weight:700; margin-bottom:2px;">{nome_sel}</div>
                    <div class="small">CNPJ: <span class="badge">{cnpj_sel}</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tabs
tab_empregados, = st.tabs(["👥 Empregados"])

with tab_empregados:
    st.markdown("### Filtros")
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        modo = st.selectbox("Modo", ["Posição", "Região"])
    with c2:
        metrica = st.selectbox("Métrica", ["Gênero", "Raça", "Idade"])

    key = (modo, metrica)
    if key not in all_dfs:
        st.warning("Arquivo correspondente não encontrado nesta pasta.")
        st.stop()

    df = all_dfs[key]
    subset = company_rows(df, cnpj_sel, nome_sel)

    if subset.empty:
        st.info("Nenhuma linha encontrada para esta empresa neste arquivo.")
        st.stop()

    # Identify if a subgroup column exists on this table
    subgroup_col = None
    for cand in ["Posicao", "Local"]:
        if cand in subset.columns and subset[cand].notna().any():
            subgroup_col = cand
            break

    # Make long format
    entity_cols = ["Nome_Companhia", "CNPJ_Companhia"]
    if subgroup_col:
        entity_cols.append(subgroup_col)
    long_df = melt_company(subset[entity_cols + find_measure_columns(subset)], entity_cols)

    # Totals on top
    total_val = long_df["Valor"].sum()
    n_categories = long_df["Categoria"].nunique()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total de funcionários (amostra)", f"{int(total_val):,}".replace(",", "."))
    m2.metric("Categorias", n_categories)
    m3.metric("Linhas origem", subset.shape[0])

    st.markdown("### Visualização")
    st.altair_chart(bar_chart(long_df, stack_by=subgroup_col), use_container_width=True)

    st.markdown("### Detalhe (tabela)")
    show_cols = ["Nome_Companhia", "CNPJ_Companhia"]
    if subgroup_col:
        show_cols.append(subgroup_col)
    pivotable = long_df[show_cols + ["Categoria", "Valor"]].copy()
    # Slightly prettier category labels
    pivotable["Categoria"] = pivotable["Categoria"].map(tidy_label)
    st.dataframe(
        pivotable.sort_values(["Categoria"] + ([subgroup_col] if subgroup_col else [])),
        use_container_width=True,
        hide_index=True
    )

# ------------- EOF -------------
