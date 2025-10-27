import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data as vega_data

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="TB Global Burden — Bubble Map", layout="wide")
alt.themes.enable("opaque")

PALETTE = {"inc":"#2E86DE","mort":"#E74C3C","prev":"#F39C12","neutral":"#7F8C8D"}

# Colonnes EXACTES du CSV (on ne modifie pas le fichier)
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

# =========================================================
# Normalisation des noms pays (pour matcher des centroïdes)
# =========================================================
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
        "Congo (Brazzaville)": "Republic of the Congo",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Myanmar (Burma)": "Myanmar",
        "Czechia": "Czech Republic",
        "Türkiye": "Turkey",
        "Eswatini": "Swaziland",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Gambia, The": "Gambia",
        "The Bahamas": "Bahamas",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Vatican": "Vatican City",
        "Republic of South Sudan": "South Sudan",
        "Palestine": "Palestinian Territories",
        "Hong Kong SAR": "Hong Kong",
        "Macao SAR": "Macau",
    }
    return fixes.get(n, n)

# =========================================================
# Centroïdes pays en mémoire (lat, lon). Liste couvrant l’essentiel.
# Si un pays manque, il sera juste sans bulle (on peut l’ajouter ensuite).
# =========================================================
COUNTRY_CENTROIDS = {
    # Amériques
    "Canada": (61.07, -107.99), "United States": (39.78, -100.45), "Mexico": (23.63, -102.55),
    "Guatemala": (15.78, -90.23), "Honduras": (14.82, -86.64), "El Salvador": (13.79, -88.90),
    "Nicaragua": (12.83, -85.00), "Costa Rica": (9.75, -84.08), "Panama": (8.54, -80.78),
    "Cuba": (21.52, -79.30), "Haiti": (19.05, -72.48), "Dominican Republic": (18.90, -70.48),
    "Jamaica": (18.12, -77.30), "Bahamas": (24.25, -76.00), "Trinidad and Tobago": (10.44, -61.24),
    "Belize": (17.20, -88.67), "Barbados": (13.19, -59.54),
    "Colombia": (3.91, -73.08), "Venezuela": (7.12, -66.18), "Guyana": (4.86, -58.93),
    "Suriname": (4.13, -55.91), "Ecuador": (-1.79, -78.18), "Peru": (-9.19, -75.02),
    "Bolivia": (-16.29, -63.59), "Chile": (-37.00, -71.00), "Argentina": (-34.00, -64.00),
    "Paraguay": (-23.44, -58.44), "Uruguay": (-32.80, -56.02),
    # Europe
    "Iceland": (64.98, -18.57), "Ireland": (53.18, -8.24), "United Kingdom": (54.04, -2.80),
    "Portugal": (39.69, -8.13), "Spain": (40.24, -3.65), "France": (46.22, 2.21), "Belgium": (50.64, 4.66),
    "Netherlands": (52.13, 5.29), "Germany": (51.17, 10.45), "Denmark": (56.14, 9.52),
    "Norway": (64.57, 11.52), "Sweden": (62.79, 16.73), "Finland": (64.50, 26.00),
    "Poland": (52.13, 19.40), "Czech Republic": (49.78, 15.50), "Austria": (47.60, 14.14),
    "Switzerland": (46.80, 8.22), "Italy": (42.79, 12.07), "Slovenia": (46.12, 14.82),
    "Croatia": (45.10, 15.20), "Bosnia and Herzegovina": (44.17, 17.79), "Serbia": (44.22, 20.78),
    "Montenegro": (42.79, 19.25), "North Macedonia": (41.60, 21.75), "Greece": (39.07, 22.95),
    "Albania": (41.14, 20.03), "Bulgaria": (42.75, 25.49), "Romania": (45.94, 24.97),
    "Hungary": (47.16, 19.50), "Slovakia": (48.67, 19.70), "Belarus": (53.70, 27.95),
    "Ukraine": (48.38, 31.17), "Moldova": (47.20, 28.47), "Lithuania": (55.34, 23.90),
    "Latvia": (56.88, 24.60), "Estonia": (58.67, 25.00), "Russia": (61.52, 105.32),
    "Turkey": (39.06, 35.18),
    # Afrique
    "Morocco": (31.79, -7.09), "Algeria": (28.04, 2.96), "Tunisia": (33.89, 9.40),
    "Libya": (27.04, 18.01), "Egypt": (26.49, 29.87),
    "Mauritania": (20.26, -10.97), "Mali": (17.57, -3.99), "Senegal": (14.36, -14.47),
    "Gambia": (13.45, -15.38), "Guinea-Bissau": (12.05, -14.67), "Guinea": (10.44, -9.31),
    "Sierra Leone": (8.56, -11.78), "Liberia": (6.45, -9.31), "Ivory Coast": (7.64, -5.55),
    "Ghana": (7.96, -1.02), "Togo": (8.53, 0.82), "Benin": (9.32, 2.31), "Burkina Faso": (12.24, -1.56),
    "Niger": (17.61, 8.08), "Nigeria": (9.08, 8.68), "Cameroon": (5.69, 12.74), "Chad": (15.36, 18.66),
    "Central African Republic": (6.61, 20.94), "Republic of the Congo": (-0.66, 15.56),
    "Democratic Republic of the Congo": (-2.88, 23.66), "Gabon": (-0.59, 11.79),
    "Equatorial Guinea": (1.61, 10.52), "Sao Tome and Principe": (0.21, 6.61),
    "South Sudan": (7.31, 30.10), "Sudan": (15.49, 29.44), "Ethiopia": (8.62, 39.60),
    "Eritrea": (15.18, 39.78), "Djibouti": (11.75, 42.59), "Somalia": (6.04, 45.33),
    "Kenya": (0.18, 37.86), "Uganda": (1.37, 32.29), "Tanzania": (-6.37, 34.89),
    "Rwanda": (-1.94, 29.88), "Burundi": (-3.36, 29.93), "Angola": (-11.20, 17.87),
    "Namibia": (-22.15, 17.20), "Botswana": (-22.33, 24.69), "South Africa": (-28.48, 24.68),
    "Lesotho": (-29.58, 28.24), "Eswatini": (-26.52, 31.47), "Zimbabwe": (-19.00, 29.15),
    "Zambia": (-13.13, 27.85), "Malawi": (-13.25, 34.30), "Mozambique": (-17.27, 35.53),
    "Madagascar": (-19.37, 46.70), "Comoros": (-11.88, 43.87), "Mauritius": (-20.25, 57.55),
    "Seychelles": (-4.68, 55.45),
    # Moyen-Orient / Asie centrale
    "Israel": (31.20, 34.86), "Lebanon": (33.92, 35.89), "Syria": (34.80, 38.98),
    "Jordan": (31.24, 36.76), "Iraq": (33.22, 43.68), "Saudi Arabia": (23.94, 45.08),
    "Yemen": (15.55, 48.52), "Oman": (20.59, 56.09), "United Arab Emirates": (24.23, 53.66),
    "Qatar": (25.30, 51.15), "Bahrain": (26.07, 50.55), "Kuwait": (29.27, 47.50),
    "Iran": (32.43, 53.69), "Afghanistan": (33.94, 67.71), "Pakistan": (29.95, 69.35),
    "Azerbaijan": (40.35, 47.70), "Armenia": (40.29, 44.94), "Georgia": (42.32, 43.37),
    "Kazakhstan": (48.16, 67.30), "Uzbekistan": (41.38, 64.57), "Turkmenistan": (39.10, 59.37),
    "Kyrgyzstan": (41.46, 74.56), "Tajikistan": (38.86, 71.27),
    # Asie du Sud / Est
    "India": (22.88, 79.80), "Sri Lanka": (7.86, 80.68), "Nepal": (28.39, 84.12),
    "Bhutan": (27.41, 90.43), "Bangladesh": (23.69, 90.35), "Maldives": (3.67, 73.54),
    "Myanmar": (19.75, 96.10), "Thailand": (15.12, 101.00), "Laos": (19.86, 102.50),
    "Cambodia": (12.69, 104.90), "Vietnam": (15.62, 106.25), "Malaysia": (4.21, 109.20),
    "Singapore": (1.35, 103.82), "Indonesia": (-2.60, 118.02), "Philippines": (12.75, 122.73),
    "Brunei": (4.52, 114.72), "Timor-Leste": (-8.79, 125.85),
    "China": (35.86, 104.19), "Mongolia": (46.86, 103.84), "Japan": (36.20, 138.25),
    "South Korea": (36.50, 127.98), "North Korea": (40.34, 127.51), "Taiwan": (23.70, 121.08),
    # Océanie
    "Australia": (-25.27, 133.77), "New Zealand": (-41.29, 174.78),
    "Papua New Guinea": (-6.31, 146.38), "Fiji": (-17.82, 178.13), "Solomon Islands": (-9.23, 160.14),
    "Vanuatu": (-15.38, 166.96), "Samoa": (-13.76, -172.10), "Tonga": (-21.18, -175.20),
    "Micronesia": (6.88, 158.22), "Palau": (7.50, 134.62), "Kiribati": (1.87, -157.36),
    "Marshall Islands": (7.12, 171.18), "Nauru": (-0.53, 166.93), "Tuvalu": (-7.11, 177.65),
}

# =========================================================
# Chargement CSV (racine OU data/), aucun renommage disque
# =========================================================
@st.cache_data(show_spinner=True)
def load_data():
    candidates = ["TB_Burden_Country.csv", os.path.join("data", "TB_Burden_Country.csv")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("CSV introuvable. Placez-le à la racine (TB_Burden_Country.csv) ou dans data/.")
        st.stop()

    raw = pd.read_csv(path)

    # Vérification colonnes attendues
    needed = list(COL.values())
    miss = [c for c in needed if c not in raw.columns]
    if miss:
        st.error(f"Colonnes manquantes dans le CSV: {miss}")
        st.stop()

    df = raw[needed].copy()

    # Types
    for k in ["year","pop","prev100k","mort100k","morthiv100k","inc100k","hiv_share_pct","cdr_pct"]:
        df[COL[k]] = pd.to_numeric(df[COL[k]], errors="coerce")

    # Colonnes auxiliaires en mémoire
    df["country_raw"]  = df[COL["country"]]
    df["country_norm"] = df[COL["country"]].apply(normalize_country_name)

    # Dérivés absolus (tooltips)
    df["prev_abs"] = (df[COL["prev100k"]] * df[COL["pop"]] / 1e5).round()
    df["inc_abs"]  = (df[COL["inc100k"]]  * df[COL["pop"]] / 1e5).round()
    df["mort_abs"] = (df[COL["mort100k"]] * df[COL["pop"]] / 1e5).round()

    # Plage années réelle (ex: 1990–2013 dans votre fichier)
    df = df.dropna(subset=[COL["year"], "country_norm"]).copy()
    df = df[(df[COL["year"]] >= 1990) & (df[COL["year"]] <= 2025)]

    return df

df = load_data()
YEARS = sorted(df[COL["year"]].unique().tolist())
LATEST = int(max(YEARS)) if YEARS else 2013

# Fond du monde (topojson intégré)
WORLD = alt.topo_feature(vega_data.world_110m.url, "countries")

def fmt_int(x):
    try: return f"{int(x):,}".replace(",", " ")
    except: return "NA"

# =========================================================
# UI: TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["Contexte", "Carte mondiale", "Pays", "Conclusion"])

# -------------------------
# 1) Contexte
# -------------------------
with tab1:
    st.title("Tuberculose: fardeau mondial visualisé")
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown(
            """
            Carte à bulles interactive du fardeau de la tuberculose par pays.
            Choisissez **Prévalence/100k**, **Incidence/100k** ou **Mortalité/100k** et explorez l’évolution par pays.
            Les tailles de cercles sont proportionnelles à la **métrique pour 100 000**.
            """
        )
    with c2:
        dfl = df[df[COL["year"]] == LATEST]
        st.metric("Cas incidents (monde)", fmt_int(dfl["inc_abs"].sum()))
        st.metric("Décès (monde)", fmt_int(dfl["mort_abs"].sum()))
        st.metric("Prévalence (monde)", fmt_int(dfl["prev_abs"].sum()))

    # Tendances mondiales pondérées
    st.subheader("Tendances mondiales (pondérées population)")
    trend = (
        df.groupby(COL["year"])
          .apply(lambda g: pd.Series({
              "Incidence/100k": np.average(g[COL["inc100k"]],  weights=g[COL["pop"]]),
              "Mortalité/100k": np.average(g[COL["mort100k"]], weights=g[COL["pop"]]),
              "Prévalence/100k": np.average(g[COL["prev100k"]], weights=g[COL["pop"]]),
          }))
          .reset_index().rename(columns={COL["year"]: "Année"})
          .melt("Année", var_name="Métrique", value_name="Valeur")
    )
    color_map = alt.Scale(domain=["Incidence/100k","Mortalité/100k","Prévalence/100k"],
                          range=[PALETTE["inc"], PALETTE["mort"], PALETTE["prev"]])
    line = alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X("Année:O"), y=alt.Y("Valeur:Q", title="Pour 100 000"),
        color=alt.Color("Métrique:N", scale=color_map),
        tooltip=["Année","Métrique", alt.Tooltip("Valeur:Q", format=".1f")]
    ).properties(height=260)
    st.altair_chart(line, use_container_width=True)

# -------------------------
# 2) Carte mondiale (BUBBLES)
# -------------------------
with tab2:
    st.header("Carte mondiale interactive")

    cf1, cf2 = st.columns([2,1])
    with cf1:
        year_sel = st.slider("Année", min_value=int(min(YEARS)), max_value=int(max(YEARS)),
                             value=LATEST, step=1)
    with cf2:
        metric_alias = st.selectbox("Métrique", options=["Prévalence/100k","Incidence/100k","Mortalité/100k"])

    # Mappage métrique -> colonnes, couleurs
    if metric_alias == "Prévalence/100k":
        metric_col = COL["prev100k"]; abs_col = "prev_abs"; color = PALETTE["prev"]
    elif metric_alias == "Incidence/100k":
        metric_col = COL["inc100k"];  abs_col = "inc_abs";  color = PALETTE["inc"]
    else:
        metric_col = COL["mort100k"]; abs_col = "mort_abs"; color = PALETTE["mort"]

    # Données année sélectionnée + centroïdes
    show = df[df[COL["year"]] == year_sel].copy()
    show["country_norm"] = show[COL["country"]].apply(normalize_country_name)
    show["lat"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[0])
    show["lon"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[1])
    show = show.dropna(subset=[metric_col, "lat", "lon"])

    # Échelle de taille robuste: entre quantile 10% et 99% pour éviter les cercles extrêmes
    q10 = np.nanpercentile(show[metric_col], 10)
    q99 = np.nanpercentile(show[metric_col], 99)
    size_scale = alt.Scale(domain=[q10, q99], range=[30, 1800])  # rayon visuel

    # Fond gris
    base_map = alt.Chart(alt.topo_feature(vega_data.world_110m.url, "countries")).mark_geoshape(
        fill="#EEEEEE", stroke="white", strokeWidth=0.25
    ).project(type="equirectangular").properties(height=520)

    # Bulles (géoprojeter lon/lat)
    bubbles = alt.Chart(show).mark_circle(opacity=0.65, stroke="white", strokeWidth=0.5, color=color).encode(
        longitude="lon:Q", latitude="lat:Q",
        size=alt.Size(f"{metric_col}:Q", title=metric_alias, scale=size_scale, legend=alt.Legend(orient="right")),
        tooltip=[
            alt.Tooltip(COL["country"], title="Pays"),
            alt.Tooltip(metric_col, title=metric_alias, format=".1f"),
            alt.Tooltip(abs_col, title="Nombre (approx.)", format=",")
        ]
    ).project(type="equirectangular").properties(height=520)

    st.altair_chart(base_map + bubbles, use_container_width=True)

    # Sélecteur pays pour l’onglet suivant
    st.session_state["country_focus"] = st.selectbox(
        "Choisir un pays pour le zoom",
        options=sorted(show[COL["country"]].dropna().unique())
    )

# -------------------------
# 3) Pays (évolution)
# -------------------------
with tab3:
    country = st.session_state.get("country_focus", None)
    if not country:
        st.info("Sélectionnez un pays dans l’onglet Carte mondiale.")
    else:
        st.header(f"Évolution — {country}")
        dpc = df[df[COL["country"]] == country].sort_values(COL["year"]).copy()
        if dpc.empty:
            st.info("Aucune donnée temporelle disponible pour ce pays.")
        else:
            long = dpc.rename(columns={
                COL["inc100k"]: "Incidence/100k",
                COL["mort100k"]: "Mortalité/100k",
                COL["prev100k"]: "Prévalence/100k",
                COL["year"]: "Année"
            })[["Année","Incidence/100k","Mortalité/100k","Prévalence/100k"]].melt(
                "Année", var_name="Métrique", value_name="Valeur"
            )
            color_map2 = alt.Scale(domain=["Incidence/100k","Mortalité/100k","Prévalence/100k"],
                                   range=[PALETTE["inc"], PALETTE["mort"], PALETTE["prev"]])
            line = alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("Année:O"),
                y=alt.Y("Valeur:Q", title="Pour 100 000"),
                color=alt.Color("Métrique:N", scale=color_map2),
                tooltip=["Année","Métrique", alt.Tooltip("Valeur:Q", format=".1f")]
            ).properties(height=320)
            st.altair_chart(line, use_container_width=True)

            # Indicateurs récents
            last = dpc.iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Incidence/100k", f"{last[COL['inc100k']]:.1f}" if pd.notna(last[COL['inc100k']]) else "NA")
            c2.metric("Mortalité/100k", f"{last[COL['mort100k']]:.1f}" if pd.notna(last[COL['mort100k']]) else "NA")
            c3.metric("Prévalence/100k", f"{last[COL['prev100k']]:.1f}" if pd.notna(last[COL['prev100k']]) else "NA")
            c4.metric("Détection des cas (%)", f"{last[COL['cdr_pct']]:.0f}" if pd.notna(last[COL['cdr_pct']]) else "NA")

# -------------------------
# 4) Conclusion
# -------------------------
with tab4:
    st.header("Enseignements clés")
    st.markdown(
        """
        - Le fardeau est **très inégal**: de grands cercles se concentrent dans un nombre limité de pays.  
        - Les tendances nationales montrent des **trajectoires divergentes** malgré des progrès mondiaux.  
        - La réduction de la **mortalité** passe par la **détection précoce** et l’intégration TB/VIH.  
        """
    )
    st.caption("Source: OMS, ‘Tuberculosis Burden by Country’. Le CSV reste inchangé; tous les calculs se font en mémoire.")
