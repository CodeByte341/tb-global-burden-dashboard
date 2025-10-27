import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data as vega_data

st.set_page_config(page_title="TB Global Burden — Data Story", layout="wide")
alt.themes.enable("opaque")

# Palette
PALETTE = {"inc":"#2E86DE","mort":"#E74C3C","prev":"#F39C12","neutral":"#7F8C8D"}

# ---- Colonnes EXACTES du CSV (aucune écriture dans le fichier)
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

@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join("data", "TB_Burden_Country.csv")
    if not os.path.exists(path):
        st.error("Placez le fichier dans data/TB_Burden_Country.csv")
        st.stop()

    raw = pd.read_csv(path)
    # Sélectionne uniquement les colonnes utiles, en gardant les noms d’origine
    df = raw[list(COL.values())].copy()

    # Construit des vues aliasées en mémoire (sans renommer dans le fichier)
    def col(k): return COL[k]

    # Types
    for k in ["year","pop","prev100k","mort100k","morthiv100k","inc100k","hiv_share_pct","cdr_pct"]:
        df[col(k)] = pd.to_numeric(df[col(k)], errors="coerce")

    # Dérivés « absolus » (nombre de cas/décès) à partir des taux pour 100k
    df["prev_abs"] = (df[col("prev100k")] * df[col("pop")] / 1e5).round()
    df["inc_abs"]  = (df[col("inc100k")]  * df[col("pop")] / 1e5).round()
    df["mort_abs"] = (df[col("mort100k")] * df[col("pop")] / 1e5).round()
    df["mort_hiv_abs"] = (df[col("morthiv100k")] * df[col("pop")] / 1e5).round()

    # Nettoyage simple
    df = df.dropna(subset=[col("country"), col("year")]).copy()
    df[col("iso3")] = df[col("iso3")].astype(str).str.upper().str.strip()
    df = df[(df[col("year")] >= 2000) & (df[col("year")] <= 2025)]

    return df

df = load_data()
YEARS = sorted(df[COL["year"]].unique().tolist())
LATEST = int(df[COL["year"]].max())

# Geo: fond de carte monde (aucun appel externe)
WORLD = alt.topo_feature(vega_data.world_110m.url, "countries")

# Utilitaires
def fmt_int(x): 
    try: return f"{int(x):,}".replace(",", " ")
    except: return "NA"

# =========================
# Onglet 1 — Contexte
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Contexte", "Carte mondiale", "Pays", "Conclusion"])

with tab1:
    st.title("Tuberculose: un fardeau mondial concentré")
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown(
            """
            La tuberculose demeure l’une des maladies infectieuses les plus meurtrières. 
            Son fardeau est **concentré**: une poignée de pays portent la majorité des nouveaux cas.
            Ce tableau de bord présente **incidence**, **mortalité** et **prévalence**, 
            avec possibilité de zoomer par pays et de comparer aux régions OMS.
            """
        )
    with colB:
        dfl = df[df[COL["year"]] == LATEST]
        st.metric("Cas incidents (monde)", fmt_int(dfl["inc_abs"].sum()))
        st.metric("Décès (monde)", fmt_int(dfl["mort_abs"].sum()))
        st.metric("Prévalence (monde)", fmt_int(dfl["prev_abs"].sum()))

    st.markdown("### Évolution mondiale (moyennes pondérées population)")
    grp = (
        df.groupby(COL["year"])
          .apply(lambda g: pd.Series({
              "inc": np.average(g[COL["inc100k"]],  weights=g[COL["pop"]]),
              "mort":np.average(g[COL["mort100k"]], weights=g[COL["pop"]]),
              "prev":np.average(g[COL["prev100k"]], weights=g[COL["pop"]]),
          }))
          .reset_index()
    )
    l1 = alt.Chart(grp).mark_line(color=PALETTE["inc"], strokeWidth=3).encode(x=alt.X(f"{COL['year']}:O", title="Année"), y=alt.Y("inc:Q", title="Pour 100 000"))
    l2 = alt.Chart(grp).mark_line(color=PALETTE["mort"], strokeDash=[6,3], strokeWidth=3).encode(x=f"{COL['year']}:O", y="mort:Q")
    l3 = alt.Chart(grp).mark_line(color=PALETTE["prev"], strokeDash=[2,2], strokeWidth=3).encode(x=f"{COL['year']}:O", y="prev:Q")
    st.altair_chart(l1 + l2 + l3, use_container_width=True)
    st.caption("Bleu: incidence, Rouge: mortalité, Orange: prévalence.")

# =========================
# Onglet 2 — Carte mondiale
# =========================
with tab2:
    st.header("Carte mondiale interactive")
    colf1, colf2 = st.columns([2,1])
    with colf1:
        year_sel = st.slider("Année", min_value=int(min(YEARS)), max_value=int(max(YEARS)), value=LATEST, step=1)
    with colf2:
        metric_key = st.selectbox(
            "Métrique",
            options=[("inc_100k", COL["inc100k"]), ("mort_100k", COL["mort100k"]), ("prev_100k", COL["prev100k"])],
            format_func=lambda t: {COL["inc100k"]:"Incidence/100k", COL["mort100k"]:"Mortalité/100k", COL["prev100k"]:"Prévalence/100k"}[t[1]]
        )
        metric_alias, metric_col = metric_key

    show = df[df[COL["year"]] == year_sel].copy()

    # Choix palette et colonne "absolu" pour le tooltip
    palette = {"inc_100k":("blues","inc_abs","Incidence/100k","Cas incidents"),
               "mort_100k":("reds","mort_abs","Mortalité/100k","Décès"),
               "prev_100k":("oranges","prev_abs","Prévalence/100k","Personnes avec TB")}
    scheme, abs_col, label_metric, label_abs = palette[metric_alias]

    # Choroplèthe relié par NOM de pays (aucun besoin d’id numérique ni d’ISO)
    # On lookup 'properties.name' du topojson avec la colonne "Country or territory name"
    choro = alt.Chart(WORLD).mark_geoshape(stroke="white", strokeWidth=0.25).encode(
        color=alt.Color(f"{metric_alias}:Q", title=label_metric, scale=alt.Scale(scheme=scheme)),
        tooltip=[alt.Tooltip("country:N", title="Pays"),
                 alt.Tooltip(f"{metric_alias}:Q", title=label_metric, format=".1f"),
                 alt.Tooltip(f"{abs_col}:Q", title=label_abs, format=",")]
    ).transform_lookup(
        lookup="properties.name",
        from_=alt.LookupData(show.rename(columns={
            COL["country"]: "country",
            metric_col: metric_alias
        }), "country", ["country", metric_alias, abs_col])
    ).project(type="equirectangular").properties(height=460)

    st.altair_chart(choro, use_container_width=True)

    # Top 20
    st.subheader("Top 20")
    top = show[[COL["country"], metric_col, abs_col]].dropna(subset=[metric_col]).copy()
    top = top.rename(columns={COL["country"]:"country", metric_col:metric_alias}).sort_values(metric_alias, ascending=False).head(20)
    bars = alt.Chart(top).mark_bar().encode(
        x=alt.X(f"{metric_alias}:Q", title=label_metric),
        y=alt.Y("country:N", sort="-x", title=None),
        color=alt.value(PALETTE["inc"] if metric_alias=="inc_100k" else (PALETTE["prev"] if metric_alias=="prev_100k" else PALETTE["mort"])),
        tooltip=["country", alt.Tooltip(metric_alias, format=".1f"), alt.Tooltip(abs_col, format=",", title=label_abs)]
    ).properties(height=22 * len(top))
    st.altair_chart(bars, use_container_width=True)

    st.session_state["country_focus"] = st.selectbox("Zoom pays", options=sorted(show[COL["country"]].dropna().unique()))

# =========================
# Onglet 3 — Pays
# =========================
with tab3:
    c = st.session_state.get("country_focus", None)
    if not c:
        st.info("Sélectionnez un pays dans l’onglet Carte mondiale.")
    else:
        st.header(f"Évolution — {c}")
        dpc = df[df[COL["country"]] == c].sort_values(COL["year"]).copy()

        base = alt.Chart(dpc).encode(x=alt.X(f"{COL['year']}:O", title="Année"))
        l_inc  = base.mark_line(color=PALETTE["inc"], strokeWidth=3).encode(y=alt.Y(f"{COL['inc100k']}:Q", title="Pour 100 000"),
                                                                            tooltip=[alt.Tooltip(COL["year"], title="Année"),
                                                                                     alt.Tooltip(COL["inc100k"], title="Incidence/100k", format=".1f")])
        l_mort = base.mark_line(color=PALETTE["mort"], strokeDash=[6,3], strokeWidth=3).encode(y=alt.Y(f"{COL['mort100k']}:Q", title=None),
                                                                                                tooltip=[alt.Tooltip(COL["mort100k"], title="Mortalité/100k", format=".1f")])
        l_prev = base.mark_line(color=PALETTE["prev"], strokeDash=[2,2], strokeWidth=3).encode(y=alt.Y(f"{COL['prev100k']}:Q", title=None),
                                                                                                tooltip=[alt.Tooltip(COL["prev100k"], title="Prévalence/100k", format=".1f")])
        st.altair_chart(l_inc + l_mort + l_prev, use_container_width=True)

        # Tuiles récentes
        last = dpc.iloc[-1]
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Incidence/100k", f"{last[COL['inc100k']]:.1f}")
        col2.metric("Mortalité/100k", f"{last[COL['mort100k']]:.1f}")
        col3.metric("Prévalence/100k", f"{last[COL['prev100k']]:.1f}")
        col4.metric("Détection des cas (%)", "NA" if np.isnan(last[COL["cdr_pct"]]) else f"{last[COL['cdr_pct']]:.0f}")

        # Slope chart 2019 → 2021 → 2023 si dispo
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

# =========================
# Onglet 4 — Conclusion
# =========================
with tab4:
    st.header("Enseignements clés")
    st.markdown(
        """
        - Le fardeau de la tuberculose reste **très inégal** et concentré dans quelques pays.  
        - Les progrès de long terme existent, mais **le choc COVID** a freiné la trajectoire.  
        - La réduction durable de la **mortalité** passe par une meilleure **détection** et la prise en charge TB/VIH.  
        """
    )
    st.caption("Source: OMS, ‘Tuberculosis Burden by Country’. Le CSV n’est jamais modifié; tous les calculs sont effectués en mémoire.")
