# app.py — Região + (Gênero/Raça/Idade) = BARRAS | Região + Empregados = MAPA (CSV wide)
import json, re
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FRE • Empregados", page_icon="👔", layout="wide")
alt.data_transformers.disable_max_rows()

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.header-box { background: #0f172a; color: white; padding: 16px 18px; border-radius: 14px; }
.badge { background: #e2e8f0; color: #0f172a; padding: 2px 8px; border-radius: 999px; font-size: 0.9rem; }
.small { opacity: 0.8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

FILE_MAP: Dict[Tuple[str, str], str] = {
    ("Posição", "Gênero"): "fre_cia_aberta_empregado_posicao_declaracao_genero_2024.csv",
    ("Posição", "Raça"):   "fre_cia_aberta_empregado_posicao_declaracao_raca_2024.csv",
    ("Posição", "Idade"):  "fre_cia_aberta_empregado_posicao_faixa_etaria_2024.csv",
    ("Região",  "Gênero"): "fre_cia_aberta_empregado_local_declaracao_genero_2024.csv",
    ("Região",  "Raça"):   "fre_cia_aberta_empregado_local_declaracao_raca_2024.csv",
    ("Região",  "Idade"):  "fre_cia_aberta_empregado_local_faixa_etaria_2024.csv",
    ("Região",  "Empregados"): "fre_cia_aberta_empregado_posicao_local_2024.csv",  # WIDE por região
}
BASE_DIR = Path.cwd()

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path, encoding=enc, sep=";", engine="python",
                dtype={"CNPJ_Companhia":"string","CNPJ_Cia":"string","CNPJ_CIA":"string","CNPJ":"string"}
            )
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"Falha ao ler {path.name}")

    rename_map = {
        "CNPJ_Cia":"CNPJ_Companhia","CNPJ_CIA":"CNPJ_Companhia","CNPJ":"CNPJ_Companhia",
        "Nome_Companhia ":"Nome_Companhia","Posição":"Posicao","Região":"Local","Regiao":"Local"
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    df.columns = [c.strip() for c in df.columns]
    if "CNPJ_Companhia" in df.columns:
        df["CNPJ_Companhia"] = df["CNPJ_Companhia"].astype("string")
    return df

@st.cache_data(show_spinner=False)
def load_all(base_dir: Path) -> dict:
    out = {}
    for key,fname in FILE_MAP.items():
        p = base_dir / fname
        if p.exists():
            out[key] = load_csv(p)
    return out

def find_measure_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("Quantidade")]
    if not cols and "Quantidade" in df.columns:
        cols = ["Quantidade"]
    return cols

def melt_company(df: pd.DataFrame, entity_cols: List[str]) -> pd.DataFrame:
    measures = find_measure_columns(df)
    if not measures: return pd.DataFrame()
    return df.melt(id_vars=entity_cols, value_vars=measures,
                   var_name="Categoria", value_name="Valor")

def tidy_label(label: str) -> str:
    label = re.sub(r"^Quantidade_+", "", label).replace("_"," ")
    if "Anos" in label and "a" in label: label = label.replace("a"," a ")
    return re.sub(r"\s+"," ",label).strip()

def company_rows(df: pd.DataFrame, cnpj: str, nome: str) -> pd.DataFrame:
    m = False
    if "CNPJ_Companhia" in df.columns and pd.notna(cnpj):
        m = (df["CNPJ_Companhia"].astype(str) == str(cnpj))
    if "Nome_Companhia" in df.columns and pd.notna(nome):
        m = m | (df["Nome_Companhia"].astype(str) == str(nome))
    return df[m].copy()

# --------- Mapa Brasil (geobr) ----------
REGION_CANON = {
    "norte":"Norte","nordeste":"Nordeste","centro-oeste":"Centro-Oeste",
    "centro oeste":"Centro-Oeste","co":"Centro-Oeste","sudeste":"Sudeste","sul":"Sul",
}
VALID_REGIONS = ["Norte","Nordeste","Centro-Oeste","Sudeste","Sul"]

@st.cache_data(show_spinner=False)
def build_regions_geojson_and_labels():
    try:
        from geobr import read_state
    except Exception as e:
        raise RuntimeError("Instale dependências do mapa: pip install geopandas geobr shapely pyproj") from e
    gdf = read_state(year=2020)[["name_region","geometry"]].dissolve(by="name_region", as_index=False).to_crs(4326)
    labels = gdf.copy()
    labels["lon"] = labels.geometry.representative_point().x
    labels["lat"] = labels.geometry.representative_point().y
    gj = json.loads(gdf.to_json())
    labels = labels.rename(columns={"name_region":"Regiao"})[["Regiao","lon","lat"]]
    return gj, labels

def normalize_region(s: str) -> str:
    if not isinstance(s,str): return s
    return REGION_CANON.get(s.strip().lower(), s.strip().title())

def ensure_all_regions(df_reg: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"Regiao": VALID_REGIONS})
    out = base.merge(df_reg, on="Regiao", how="left").fillna({c:0 for c in df_reg.columns if c!="Regiao"})
    if "Valor" in out.columns:
        out["Valor"] = pd.to_numeric(out["Valor"], errors="coerce").fillna(0)
    return out

def make_region_choropleth(df_reg: pd.DataFrame, value_col="Valor"):
    gj, lbl = build_regions_geojson_and_labels()
    df_plot = ensure_all_regions(df_reg)[["Regiao", value_col]]

    fig = px.choropleth(
        df_plot, geojson=gj, featureidkey="properties.name_region",
        locations="Regiao", color=value_col, color_continuous_scale="Blues",
        projection="mercator", labels={value_col: "Quantidade"},
    )
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    lbl_plot = lbl.merge(df_plot, on="Regiao", how="left")
    fig.add_trace(go.Scattergeo(
        lon=lbl_plot["lon"], lat=lbl_plot["lat"],
        text=[f"{int(v):d}" for v in lbl_plot[value_col].fillna(0)],
        mode="text", textfont=dict(size=14), hoverinfo="skip", showlegend=False
    ))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=520)
    return fig

# ---------- Carregamento ----------
all_dfs = load_all(BASE_DIR)
def get_company_index(all_dfs: dict) -> pd.DataFrame:
    frames = []
    for df in all_dfs.values():
        if {"CNPJ_Companhia","Nome_Companhia"}.issubset(df.columns):
            frames.append(df[["CNPJ_Companhia","Nome_Companhia"]].copy())
    if not frames: return pd.DataFrame(columns=["CNPJ_Companhia","Nome_Companhia"])
    return (pd.concat(frames, ignore_index=True)
            .dropna(subset=["Nome_Companhia"]).drop_duplicates()
            .sort_values("Nome_Companhia").reset_index(drop=True))

companies = get_company_index(all_dfs)
if companies.empty: st.stop()

# ---------- UI ----------
st.title("📊 FRE – Empregados por Companhia")
st.subheader("1) Escolha a companhia")
col1, col2 = st.columns([2,3])
with col1:
    search = st.text_input("Buscar por nome ou CNPJ","")
view = companies.copy()
if search.strip():
    s = search.lower().strip()
    view = view[view.apply(lambda r: s in str(r["Nome_Companhia"]).lower() or s in str(r["CNPJ_Companhia"]).lower(), axis=1)]
options = view.assign(_opt=view["Nome_Companhia"] + "  ·  " + view["CNPJ_Companhia"].fillna(""))._opt.tolist()
pick = st.selectbox("Selecione a empresa", options, index=0)
chosen = view.iloc[options.index(pick)] if options else view.iloc[0]
cnpj_sel, nome_sel = chosen["CNPJ_Companhia"], chosen["Nome_Companhia"]

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("2) Página da Companhia")
st.markdown(f"""
<div class="header-box">
  <div style="display:flex; gap:18px; align-items:center;">
    <div style="width:56px;height:56px;border-radius:14px;background:#334155;display:flex;align-items:center;justify-content:center;font-weight:700;">
      {(nome_sel[:2] if isinstance(nome_sel,str) else '??').upper()}
    </div>
    <div>
      <div style="font-size:1.35rem;font-weight:700;margin-bottom:2px;">{nome_sel}</div>
      <div class="small">CNPJ: <span class="badge">{cnpj_sel}</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_empregados, = st.tabs(["👥 Empregados"])

with tab_empregados:
    st.markdown("### Filtros")
    c1, c2, _ = st.columns([1,1,3])
    with c1:
        modo = st.selectbox("Modo", ["Posição","Região"])
    with c2:
        metrica_opts = ["Gênero","Raça","Idade"] if modo=="Posição" else ["Empregados","Gênero","Raça","Idade"]
        metrica = st.selectbox("Métrica", metrica_opts)

    key = (modo, metrica)
    if key not in all_dfs:
        st.warning("Arquivo correspondente não encontrado.")
        st.stop()

    df = all_dfs[key]
    subset = company_rows(df, cnpj_sel, nome_sel)
    if subset.empty:
        st.info("Nenhuma linha encontrada para esta empresa neste arquivo.")
        st.stop()

    subgroup_col = None
    for cand in ["Posicao","Local"]:
        if cand in subset.columns and subset[cand].notna().any():
            subgroup_col = cand
            break

    entity_cols = ["Nome_Companhia","CNPJ_Companhia"]
    if subgroup_col: entity_cols.append(subgroup_col)
    long_df = melt_company(subset[entity_cols + find_measure_columns(subset)], entity_cols)

    total_val = long_df["Valor"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total de funcionários (amostra)", f"{int(total_val):,}".replace(",","."))
    m2.metric("Categorias", long_df["Categoria"].nunique())
    m3.metric("Linhas origem", subset.shape[0])

    st.markdown("### Visualização")

    # ------- Região -------
    if modo == "Região":
        if metrica == "Empregados":
            # CSV wide: colunas Quantidade_<Regiao> → melt -> Regiao/Valor; somar por Regiao
            def cat_to_regiao(cat: str) -> str:
                # "Quantidade_Centro_Oeste" -> "Centro-Oeste"; "Quantidade_Exterior" -> "Exterior"
                name = re.sub(r"^Quantidade_","",cat or "", flags=re.I)
                name = name.replace("_","-").title().replace("De","de").replace("Da","da").replace("Do","do")
                # garantir hifen do Centro-Oeste
                name = name.replace("Centro-Oeste","Centro-Oeste")
                return name

            df_reg = (long_df.assign(Regiao=long_df["Categoria"].map(cat_to_regiao))
                               .groupby("Regiao", as_index=False)["Valor"].sum())

            # Mapa: apenas 5 regiões do Brasil
            df_reg_map = df_reg[df_reg["Regiao"].isin(VALID_REGIONS)].copy()
            try:
                fig = make_region_choropleth(df_reg_map, value_col="Valor")
                st.plotly_chart(fig, use_container_width=True)
            except RuntimeError as e:
                st.error(str(e))

            # Tabela detalhada (inclui Exterior, se existir)
            tbl = df_reg.rename(columns={"Regiao":"Região"}).copy()
            st.markdown("### Detalhe (tabela)")
            st.dataframe(tbl.sort_values("Região"), use_container_width=True, hide_index=True)

        else:
            # Região + (Gênero/Raça/Idade) = BARRAS (usa 'Local' se existir)
            d = long_df.copy()
            d["Categoria"] = d["Categoria"].map(tidy_label)
            if "Local" in d.columns:
                chart = alt.Chart(d).mark_bar().encode(
                    x=alt.X("Categoria:N", sort="-y", title="Categoria"),
                    y=alt.Y("sum(Valor):Q", title="Quantidade"),
                    color=alt.Color("Local:N", title="Região"),
                    tooltip=["Categoria:N","Local:N",alt.Tooltip("sum(Valor):Q",title="Qtd")]
                ).properties(height=380)
            else:
                # fallback: sem 'Local', apenas barras por categoria
                chart = alt.Chart(d).mark_bar().encode(
                    x=alt.X("Categoria:N", sort="-y", title="Categoria"),
                    y=alt.Y("Valor:Q", title="Quantidade"),
                    tooltip=["Categoria:N", alt.Tooltip("Valor:Q", title="Qtd")]
                ).properties(height=380)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### Detalhe (tabela)")
            show_cols = ["Nome_Companhia","CNPJ_Companhia"] + (["Local"] if "Local" in d.columns else [])
            t = d[show_cols + ["Categoria","Valor"]].copy()
            t["Categoria"] = t["Categoria"].map(tidy_label)
            st.dataframe(t.sort_values(["Categoria"] + (["Local"] if "Local" in d.columns else [])),
                         use_container_width=True, hide_index=True)

    # ------- Posição -------
    else:
        d = long_df.copy()
        d["Categoria"] = d["Categoria"].map(tidy_label)
        if "Posicao" in d.columns:
            chart = alt.Chart(d).mark_bar().encode(
                x=alt.X("Categoria:N", sort="-y", title="Categoria"),
                y=alt.Y("sum(Valor):Q", title="Quantidade"),
                color=alt.Color("Posicao:N", title="Posição"),
                tooltip=["Categoria:N","Posicao:N",alt.Tooltip("sum(Valor):Q",title="Qtd")]
            ).properties(height=380)
        else:
            chart = alt.Chart(d).mark_bar().encode(
                x=alt.X("Categoria:N", sort="-y", title="Categoria"),
                y=alt.Y("Valor:Q", title="Quantidade"),
                tooltip=["Categoria:N", alt.Tooltip("Valor:Q", title="Qtd")]
            ).properties(height=380)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Detalhe (tabela)")
        show = ["Nome_Companhia","CNPJ_Companhia"] + (["Posicao"] if "Posicao" in d.columns else [])
        t = d[show + ["Categoria","Valor"]].copy()
        st.dataframe(t.sort_values(["Categoria"] + (["Posicao"] if "Posicao" in d.columns else [])),
                     use_container_width=True, hide_index=True)
