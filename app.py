import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data as vega_data

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="TB Global Burden — Data Story", layout="wide")
alt.themes.enable("opaque")

PALETTE = {"inc":"#2E86DE","mort":"#E74C3C","prev":"#F39C12","neutral":"#7F8C8D"}

# Colonnes EXACTES du CSV (aucune modification sur disque)
COL = {
    "country": "Country or territory name",
    "iso3": "ISO 3-character country/territory code",
    "region": "Region",
    "year": "Year",
    "pop": "Estimated total population number",
    "prev100k": "Estimated prevalence of TB (all forms) per 100 000 population",
    "mort100k": "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
    "morthiv100k": "Estimated mortality of TB cases who are HIV-positive, per 100 000 population",
    "inc100k": "Estimated incidence (all forms) per 100 000 population",
    "hiv_share_pct": "Estimated HIV in incident TB (percent)",
    "cdr_pct": "Case detection rate (all forms), percent",
}

# -------------------------------------------------
# Normalisation des noms de pays (en mémoire)
# -------------------------------------------------
def normalize_country_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    n = name.strip()
    fixes = {
        "Côte d'Ivoire": "Ivory Coast",
        "Viet Nam": "Vietnam",
        "Russian Federation": "Russia",
        "United States of America": "United States",
        "United Republic of Tanzania": "Tanzania",
        "Syrian Arab Republic": "Syria",
        "Cabo Verde": "Cape Verde",
        "Lao People's Democratic Republic": "Laos",
        "Iran (Islamic Republic of)": "Iran",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Micronesia (Federated States of)": "Federated States of Micronesia",
        "Republic of Moldova": "Moldova",
        "Republic of Korea": "South Korea",
        "Democratic People's Republic of Korea": "North Korea",
        "Timor-Leste": "East Timor",
        "Congo": "Republic of the Congo",
        "Palestine": "Palestinian Territories",
        "Czechia": "Czech Republic",
        "Türkiye": "Turkey",
        "Eswatini": "Swaziland",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Congo (Brazzaville)": "Republic of the Congo",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Myanmar (Burma)": "Myanmar",
        "Gambia, The": "Gambia",
        "The Bahamas": "Bahamas"
    }
    return fixes.get(n, n)

# -------------------------------------------------
# Chargement des données (racine OU data/)
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    candidates = [
        "TB_Burden_Country.csv",
        os.path.join("data", "TB_Burden_Country.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("Fichier CSV introuvable. Placez-le à la racine (TB_Burden_Country.csv) "
                 "ou dans data/TB_Burden_Country.csv.")
        st.stop()

    raw = pd.read_csv(path)

    # Sélection utile en gardant les NOMS D’ORIGINE
    keep = list(COL.values())
    missing = [c for c in keep if c not in raw.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le CSV: {missing}")
        st.stop()

    df = raw[keep].copy()

    # Types numériques
    for k in ["year","pop","prev100k","mort100k","morthiv100k","inc100k","hiv_share_pct","cdr_pct"]:
        df[COL[k]] = pd.to_numeric(df[COL[k]], errors="coerce")

    # Colonnes auxiliaires en mémoire
    df["country_raw"]  = df[COL["country"]]
    df["country_norm"] = df[COL["country"]].apply(normalize_country_name)

    # Dérivés absolus (pour tooltips)
    df["prev_abs"] = (df[COL["prev100k"]] * df[COL["pop"]] / 1e5).round()
    df["inc_abs"]  = (df[COL["inc100k"]]  * df[COL["pop"]] / 1e5).round()
    df["mort_abs"] = (df[COL["mort100k"]] * df[COL["pop"]] / 1e5).round()
    df["mort_hiv_abs"] = (df[COL["morthiv100k"]] * df[COL["pop"]] / 1e5).round()

    # Filtrage plage années présente (ex: 1990–2013 dans votre fichier)
    df = df.dropna(subset=[COL["year"], "country_norm"]).copy()
    df = df[(df[COL["year"]] >= 1990) & (df[COL["year"]] <= 2025)]

    return df

df = load_data()
YEARS = sorted(df[COL["year"]].unique().tolist())
LATEST = int(max(YEARS)) if YEARS else 2013

# Fond de carte intégré (aucun appel externe)
WORLD = alt.topo_feature(vega_data.world_110m.url, "countries")

def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return "NA"

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Contexte", "Carte mondiale", "Pays", "Conclusion"])

# =========================
# 1) CONTEXTE
# =========================
with tab1:
    st.title("Tuberculose: un fardeau mondial concentré")
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown(
            """
            La tuberculose demeure l’une des maladies infectieuses les plus meurtrières. 
            Son fardeau est **très inégal**: une poignée de pays regroupent la majorité des cas.
            
            Ce tableau de bord propose une lecture **incidence / mortalité / prévalence** par pays et par année, 
            une **carte interactive**, un **zoom pays** et des **messages d’action**.
            """
        )
    with c2:
        dfl = df[df[COL["year"]] == LATEST]
        st.metric("Cas incidents (monde)", fmt_int(dfl["inc_abs"].sum()))
        st.metric("Décès (monde)", fmt_int(dfl["mort_abs"].sum()))
        st.metric("Prévalence (monde)", fmt_int(dfl["prev_abs"].sum()))

    st.markdown("### Tendances pondérées (monde)")
    trend = (
        df.groupby(COL["year"])
          .apply(lambda g: pd.Series({
              "inc":  np.average(g[COL["inc100k"]],  weights=g[COL["pop"]]),
              "mort": np.average(g[COL["mort100k"]], weights=g[COL["pop"]]),
              "prev": np.average(g[COL["prev100k"]], weights=g[COL["pop"]]),
          }))
          .reset_index()
    )
    l1 = alt.Chart(trend).mark_line(color=PALETTE["inc"], strokeWidth=3).encode(
        x=alt.X(f"{COL['year']}:O", title="Année"),
        y=alt.Y("inc:Q", title="Pour 100 000"),
        tooltip=[alt.Tooltip(COL["year"], title="Année"), alt.Tooltip("inc:Q", format=".1f")]
    )
    l2 = alt.Chart(trend).mark_line(color=PALETTE["mort"], strokeDash=[6,3], strokeWidth=3).encode(
        x=f"{COL['year']}:O",
        y=alt.Y("mort:Q", title=None),
        tooltip=[alt.Tooltip("mort:Q", format=".1f", title="Mortalité/100k")]
    )
    l3 = alt.Chart(trend).mark_line(color=PALETTE["prev"], strokeDash=[2,2], strokeWidth=3).encode(
        x=f"{COL['year']}:O",
        y=alt.Y("prev:Q", title=None),
        tooltip=[alt.Tooltip("prev:Q", format=".1f", title="Prévalence/100k")]
    )
    st.altair_chart(l1 + l2 + l3, use_container_width=True)
    st.caption("Bleu: incidence, Rouge: mortalité, Orange: prévalence. Pondération par la population.")

# =========================
# 2) CARTE MONDIALE
# =========================
with tab2:
    st.header("Carte mondiale interactive")

    cf1, cf2 = st.columns([2,1])
    with cf1:
        year_sel = st.slider("Année", min_value=int(min(YEARS)), max_value=int(max(YEARS)),
                             value=LATEST, step=1)
    with cf2:
        key = st.selectbox(
            "Métrique",
            options=[("inc_100k", COL["inc100k"]), ("mort_100k", COL["mort100k"]), ("prev_100k", COL["prev100k"])],
            format_func=lambda t: {COL["inc100k"]:"Incidence/100k", COL["mort100k"]:"Mortalité/100k", COL["prev100k"]:"Prévalence/100k"}[t[1]]
        )
        metric_alias, metric_col = key

    # Sous-ensemble année + normalisation noms
    show = df[df[COL["year"]] == year_sel].copy()
    show["country_norm"] = show[COL["country"]].apply(normalize_country_name)
    show["country_raw"]  = show[COL["country"]]

    # Palette et colonne absolue pour tooltips
    palette = {
        "inc_100k": ("blues", "inc_abs",  "Incidence/100k",  "Cas incidents"),
        "mort_100k":("reds",  "mort_abs", "Mortalité/100k",  "Décès"),
        "prev_100k":("oranges","prev_abs","Prévalence/100k","Personnes avec TB"),
    }
    scheme, abs_col, label_metric, label_abs = palette[metric_alias]

    # Couche de fond grise (toujours visible)
    base_map = alt.Chart(WORLD).mark_geoshape(
        fill="#EEEEEE", stroke="white", strokeWidth=0.25
    ).project(type="equirectangular").properties(height=480)

    # Choroplèthe joint par NOM normalisé
    choro = alt.Chart(WORLD).mark_geoshape(stroke="white", strokeWidth=0.25).encode(
        color=alt.Color(f"{metric_alias}:Q", title=label_metric, scale=alt.Scale(scheme=scheme)),
        tooltip=[
            alt.Tooltip("country_norm:N", title="Pays"),
            alt.Tooltip(f"{metric_alias}:Q", title=label_metric, format=".1f"),
            alt.Tooltip(f"{abs_col}:Q", title=label_abs, format=",")
        ]
    ).transform_lookup(
        lookup="properties.name",
        from_=alt.LookupData(
            show.rename(columns={metric_col: metric_alias})[
                ["country_norm", metric_alias, abs_col]
            ],
            "country_norm",
            ["country_norm", metric_alias, abs_col]
        )
    ).project(type="equirectangular").properties(height=480)

    st.altair_chart(base_map + choro, use_container_width=True)

    # Top 20 bar chart
    st.subheader("Top 20")
    top = (
        show.rename(columns={metric_col: metric_alias})
            .dropna(subset=[metric_alias])
            .sort_values(metric_alias, ascending=False)
            .head(20)
            .copy()
    )
    top["country"] = top["country_raw"]
    bars = alt.Chart(top).mark_bar().encode(
        x=alt.X(f"{metric_alias}:Q", title=label_metric),
        y=alt.Y("country:N", sort="-x", title=None),
        color=alt.value(PALETTE["inc"] if metric_alias=="inc_100k" else (PALETTE["prev"] if metric_alias=="prev_100k" else PALETTE["mort"])),
        tooltip=["country", alt.Tooltip(metric_alias, format=".1f"), alt.Tooltip(abs_col, title=label_abs, format=",")]
    ).properties(height=22 * len(top))
    st.altair_chart(bars, use_container_width=True)

    # Sélection pays pour l’onglet suivant
    st.session_state["country_focus"] = st.selectbox(
        "Zoom pays",
        options=sorted(show["country_raw"].dropna().unique()),
        index=0 if len(show) == 0 else 0
    )

# =========================
# 3) PAYS
# =========================
with tab3:
    country = st.session_state.get("country_focus", None)
    if not country:
        st.info("Sélectionnez un pays dans l’onglet Carte mondiale.")
    else:
        st.header(f"Évolution — {country}")
        dpc = df[df[COL["country"]] == country].sort_values(COL["year"]).copy()

        base = alt.Chart(dpc).encode(x=alt.X(f"{COL['year']}:O", title="Année"))
        l_inc = base.mark_line(color=PALETTE["inc"], strokeWidth=3).encode(
            y=alt.Y(f"{COL['inc100k']}:Q", title="Pour 100 000"),
            tooltip=[alt.Tooltip(COL["year"], title="Année"), alt.Tooltip(COL["inc100k"], title="Incidence/100k", format=".1f")]
        )
        l_mort = base.mark_line(color=PALETTE["mort"], strokeDash=[6,3], strokeWidth=3).encode(
            y=alt.Y(f"{COL['mort100k']}:Q", title=None),
            tooltip=[alt.Tooltip(COL["mort100k"], title="Mortalité/100k", format=".1f")]
        )
        l_prev = base.mark_line(color=PALETTE["prev"], strokeDash=[2,2], strokeWidth=3).encode(
            y=alt.Y(f"{COL['prev100k']}:Q", title=None),
            tooltip=[alt.Tooltip(COL["prev100k"], title="Prévalence/100k", format=".1f")]
        )
        st.altair_chart(l_inc + l_mort + l_prev, use_container_width=True)

        # Tuiles dernière année
        last = dpc.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Incidence/100k", f"{last[COL['inc100k']]:.1f}" if pd.notna(last[COL["inc100k"]]) else "NA")
        c2.metric("Mortalité/100k", f"{last[COL['mort100k']]:.1f}" if pd.notna(last[COL["mort100k"]]) else "NA")
        c3.metric("Prévalence/100k", f"{last[COL['prev100k']]:.1f}" if pd.notna(last[COL["prev100k"]]) else "NA")
        c4.metric("Détection des cas (%)", f"{last[COL['cdr_pct']]:.0f}" if pd.notna(last[COL["cdr_pct"]]) else "NA")

        # Slope chart 2019→2021→2023 si dispo (selon votre période réelle 1990–2013, ce sera souvent indisponible)
        years = [y for y in [2019, 2021, 2023] if y in dpc[COL["year"]].unique()]
        if len(years) >= 2:
            sl = dpc[dpc[COL["year"]].isin(years)][[COL["year"], COL["inc100k"]]].rename(
                columns={COL["year"]:"year", COL["inc100k"]:"inc_100k"})
            slope = alt.Chart(sl).mark_line(point=True, color=PALETTE["inc"], strokeWidth=3).encode(
                x=alt.X("year:O", title=None), y=alt.Y("inc_100k:Q", title="Incidence/100k"),
                tooltip=[alt.Tooltip("year:O"), alt.Tooltip("inc_100k:Q", format=".1f")]
            ).properties(height=220)
            st.subheader("Choc COVID et récupération")
            st.altair_chart(slope, use_container_width=True)
        else:
            st.caption("Années clés pour COVID non présentes dans cette série (période principale 1990–2013).")

# =========================
# 4) CONCLUSION
# =========================
with tab4:
    st.header("Enseignements clés")
    st.markdown(
        """
        - Le fardeau de la tuberculose est **fortement concentré** dans quelques pays.  
        - Les tendances de long terme montrent des progrès, **hétérogènes selon les régions**.  
        - La réduction durable de la **mortalité** passe par la **détection précoce**, 
          un **accès continu** au traitement, et l’intégration TB/VIH.  
        """
    )
    st.caption("Source: OMS, ‘Tuberculosis Burden by Country’. Le CSV n’est jamais modifié; tous les calculs sont effectués en mémoire.")
