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
st.set_page_config(page_title="TB Global Burden — Quantités absolues", layout="wide")
alt.themes.enable("opaque")
PALETTE = {"inc":"#2E86DE","mort":"#E74C3C","prev":"#F39C12","neutral":"#7F8C8D"}

# Colonnes EXACTES du CSV (aucune modification disque)
COL = {
    "country": "Country or territory name",
    "iso3": "ISO 3-character country/territory code",
    "region": "Region",
    "year": "Year",
    "pop": "Estimated total population number",

    # ABSOLUES (on se base uniquement sur elles)
    "prev_abs": "Estimated prevalence of TB (all forms)",
    "prev_abs_lo": "Estimated prevalence of TB (all forms), low bound",
    "prev_abs_hi": "Estimated prevalence of TB (all forms), high bound",

    "deaths_abs": "Estimated number of deaths from TB (all forms, excluding HIV)",
    "deaths_abs_lo": "Estimated number of deaths from TB (all forms, excluding HIV), low bound",
    "deaths_abs_hi": "Estimated number of deaths from TB (all forms, excluding HIV), high bound",

    "inc_abs": "Estimated number of incident cases (all forms)",
    "inc_abs_lo": "Estimated number of incident cases (all forms), low bound",
    "inc_abs_hi": "Estimated number of incident cases (all forms), high bound",

    # TB/VIH absolu
    "tbhiv_abs": "Estimated incidence of TB cases who are HIV-positive",
    "tbhiv_abs_lo": "Estimated incidence of TB cases who are HIV-positive, low bound",
    "tbhiv_abs_hi": "Estimated incidence of TB cases who are HIV-positive, high bound",

    # Autres champs gardés tels quels (non utilisés pour les graphes principaux)
    "cdr_pct": "Case detection rate (all forms), percent",
}

# =========================================================
# Normalisation noms pays (pour centroids)
# =========================================================
def normalize_country_name(name: str) -> str:
    if not isinstance(name, str): return name
    n = name.strip()
    fixes = {
        "Côte d'Ivoire":"Ivory Coast","Viet Nam":"Vietnam","Russian Federation":"Russia",
        "United States of America":"United States","United Republic of Tanzania":"Tanzania",
        "Syrian Arab Republic":"Syria","Cabo Verde":"Cape Verde","Lao People's Democratic Republic":"Laos",
        "Iran (Islamic Republic of)":"Iran","Bolivia (Plurinational State of)":"Bolivia",
        "Venezuela (Bolivarian Republic of)":"Venezuela","Micronesia (Federated States of)":"Micronesia",
        "Republic of Moldova":"Moldova","Republic of Korea":"South Korea",
        "Democratic People's Republic of Korea":"North Korea","Timor-Leste":"East Timor",
        "Congo (Brazzaville)":"Republic of the Congo","Congo (Kinshasa)":"Democratic Republic of the Congo",
        "Congo":"Republic of the Congo","Myanmar (Burma)":"Myanmar","Czechia":"Czech Republic",
        "Türkiye":"Turkey","Eswatini":"Swaziland","São Tomé and Príncipe":"Sao Tome and Principe",
        "Gambia, The":"Gambia","The Bahamas":"Bahamas",
        "United Kingdom of Great Britain and Northern Ireland":"United Kingdom",
        "Republic of South Sudan":"South Sudan","Palestine":"Palestinian Territories",
        "Hong Kong SAR":"Hong Kong","Macao SAR":"Macau",
    }
    return fixes.get(n, n)

# =========================================================
# Centroïdes pays (lat, lon) — couverture large
# =========================================================
COUNTRY_CENTROIDS = {
    # Amériques
    "Canada":(61.07,-107.99), "United States":(39.78,-100.45), "Mexico":(23.63,-102.55),
    "Guatemala":(15.78,-90.23), "Honduras":(14.82,-86.64), "El Salvador":(13.79,-88.90),
    "Nicaragua":(12.83,-85.00), "Costa Rica":(9.75,-84.08), "Panama":(8.54,-80.78),
    "Cuba":(21.52,-79.30), "Haiti":(19.05,-72.48), "Dominican Republic":(18.90,-70.48),
    "Jamaica":(18.12,-77.30), "Bahamas":(24.25,-76.00), "Trinidad and Tobago":(10.44,-61.24),
    "Belize":(17.20,-88.67), "Barbados":(13.19,-59.54),
    "Colombia":(3.91,-73.08), "Venezuela":(7.12,-66.18), "Guyana":(4.86,-58.93),
    "Suriname":(4.13,-55.91), "Ecuador":(-1.79,-78.18), "Peru":(-9.19,-75.02),
    "Bolivia":(-16.29,-63.59), "Chile":(-37.00,-71.00), "Argentina":(-34.00,-64.00),
    "Paraguay":(-23.44,-58.44), "Uruguay":(-32.80,-56.02),
    # Europe
    "Iceland":(64.98,-18.57), "Ireland":(53.18,-8.24), "United Kingdom":(54.04,-2.80),
    "Portugal":(39.69,-8.13), "Spain":(40.24,-3.65), "France":(46.22,2.21), "Belgium":(50.64,4.66),
    "Netherlands":(52.13,5.29), "Germany":(51.17,10.45), "Denmark":(56.14,9.52),
    "Norway":(64.57,11.52), "Sweden":(62.79,16.73), "Finland":(64.50,26.00),
    "Poland":(52.13,19.40), "Czech Republic":(49.78,15.50), "Austria":(47.60,14.14),
    "Switzerland":(46.80,8.22), "Italy":(42.79,12.07), "Slovenia":(46.12,14.82),
    "Croatia":(45.10,15.20), "Bosnia and Herzegovina":(44.17,17.79), "Serbia":(44.22,20.78),
    "Montenegro":(42.79,19.25), "North Macedonia":(41.60,21.75), "Greece":(39.07,22.95),
    "Albania":(41.14,20.03), "Bulgaria":(42.75,25.49), "Romania":(45.94,24.97),
    "Hungary":(47.16,19.50), "Slovakia":(48.67,19.70), "Belarus":(53.70,27.95),
    "Ukraine":(48.38,31.17), "Moldova":(47.20,28.47), "Lithuania":(55.34,23.90),
    "Latvia":(56.88,24.60), "Estonia":(58.67,25.00), "Russia":(61.52,105.32),
    "Turkey":(39.06,35.18),
    # Afrique
    "Morocco":(31.79,-7.09), "Algeria":(28.04,2.96), "Tunisia":(33.89,9.40),
    "Libya":(27.04,18.01), "Egypt":(26.49,29.87),
    "Mauritania":(20.26,-10.97), "Mali":(17.57,-3.99), "Senegal":(14.36,-14.47),
    "Gambia":(13.45,-15.38), "Guinea-Bissau":(12.05,-14.67), "Guinea":(10.44,-9.31),
    "Sierra Leone":(8.56,-11.78), "Liberia":(6.45,-9.31), "Ivory Coast":(7.64,-5.55),
    "Ghana":(7.96,-1.02), "Togo":(8.53,0.82), "Benin":(9.32,2.31), "Burkina Faso":(12.24,-1.56),
    "Niger":(17.61,8.08), "Nigeria":(9.08,8.68), "Cameroon":(5.69,12.74), "Chad":(15.36,18.66),
    "Central African Republic":(6.61,20.94), "Republic of the Congo":(-0.66,15.56),
    "Democratic Republic of the Congo":(-2.88,23.66), "Gabon":(-0.59,11.79),
    "Equatorial Guinea":(1.61,10.52), "Sao Tome and Principe":(0.21,6.61),
    "South Sudan":(7.31,30.10), "Sudan":(15.49,29.44), "Ethiopia":(8.62,39.60),
    "Eritrea":(15.18,39.78), "Djibouti":(11.75,42.59), "Somalia":(6.04,45.33),
    "Kenya":(0.18,37.86), "Uganda":(1.37,32.29), "Tanzania":(-6.37,34.89),
    "Rwanda":(-1.94,29.88), "Burundi":(-3.36,29.93), "Angola":(-11.20,17.87),
    "Namibia":(-22.15,17.20), "Botswana":(-22.33,24.69), "South Africa":(-28.48,24.68),
    "Lesotho":(-29.58,28.24), "Swaziland":(-26.52,31.47), "Zimbabwe":(-19.00,29.15),
    "Zambia":(-13.13,27.85), "Malawi":(-13.25,34.30), "Mozambique":(-17.27,35.53),
    "Madagascar":(-19.37,46.70), "Comoros":(-11.88,43.87), "Mauritius":(-20.25,57.55),
    "Seychelles":(-4.68,55.45),
    # Moyen-Orient / Asie centrale
    "Israel":(31.20,34.86), "Lebanon":(33.92,35.89), "Syria":(34.80,38.98),
    "Jordan":(31.24,36.76), "Iraq":(33.22,43.68), "Saudi Arabia":(23.94,45.08),
    "Yemen":(15.55,48.52), "Oman":(20.59,56.09), "United Arab Emirates":(24.23,53.66),
    "Qatar":(25.30,51.15), "Bahrain":(26.07,50.55), "Kuwait":(29.27,47.50),
    "Iran":(32.43,53.69), "Afghanistan":(33.94,67.71), "Pakistan":(29.95,69.35),
    "Azerbaijan":(40.35,47.70), "Armenia":(40.29,44.94), "Georgia":(42.32,43.37),
    "Kazakhstan":(48.16,67.30), "Uzbekistan":(41.38,64.57), "Turkmenistan":(39.10,59.37),
    "Kyrgyzstan":(41.46,74.56), "Tajikistan":(38.86,71.27),
    # Asie du Sud / Est
    "India":(22.88,79.80), "Sri Lanka":(7.86,80.68), "Nepal":(28.39,84.12),
    "Bhutan":(27.41,90.43), "Bangladesh":(23.69,90.35), "Maldives":(3.67,73.54),
    "Myanmar":(19.75,96.10), "Thailand":(15.12,101.00), "Laos":(19.86,102.50),
    "Cambodia":(12.69,104.90), "Vietnam":(15.62,106.25), "Malaysia":(4.21,109.20),
    "Singapore":(1.35,103.82), "Indonesia":(-2.60,118.02), "Philippines":(12.75,122.73),
    "Brunei":(4.52,114.72), "East Timor":(-8.79,125.85),
    "China":(35.86,104.19), "Mongolia":(46.86,103.84), "Japan":(36.20,138.25),
    "South Korea":(36.50,127.98), "North Korea":(40.34,127.51), "Taiwan":(23.70,121.08),
    # Océanie
    "Australia":(-25.27,133.77), "New Zealand":(-41.29,174.78),
    "Papua New Guinea":(-6.31,146.38), "Fiji":(-17.82,178.13), "Solomon Islands":(-9.23,160.14),
    "Vanuatu":(-15.38,166.96), "Samoa":(-13.76,-172.10), "Tonga":(-21.18,-175.20),
    "Micronesia":(6.88,158.22), "Palau":(7.50,134.62), "Kiribati":(1.87,-157.36),
    "Marshall Islands":(7.12,171.18), "Nauru":(-0.53,166.93), "Tuvalu":(-7.11,177.65),
}

# =========================================================
# Chargement CSV (racine OU data/)
# =========================================================
@st.cache_data(show_spinner=True)
def load_data():
    candidates = ["TB_Burden_Country.csv", os.path.join("data","TB_Burden_Country.csv")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("CSV introuvable. Placez-le à la racine ou dans data/."); st.stop()

    raw = pd.read_csv(path, sep=",")
    needed = list(COL.values())
    miss = [c for c in needed if c not in raw.columns]
    if miss:
        st.error(f"Colonnes manquantes dans le CSV: {miss}"); st.stop()

    df = raw[needed].copy()

    # Types
    num_cols = [COL["year"], COL["pop"],
                COL["prev_abs"], COL["prev_abs_lo"], COL["prev_abs_hi"],
                COL["deaths_abs"], COL["deaths_abs_lo"], COL["deaths_abs_hi"],
                COL["inc_abs"], COL["inc_abs_lo"], COL["inc_abs_hi"],
                COL["tbhiv_abs"], COL["tbhiv_abs_lo"], COL["tbhiv_abs_hi"],
                COL["cdr_pct"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Auxiliaires
    df["country_raw"]  = df[COL["country"]]
    df["country_norm"] = df[COL["country"]].apply(normalize_country_name)

    # Plage années réelle
    df = df.dropna(subset=[COL["year"], "country_norm"]).copy()
    df = df[(df[COL["year"]] >= 1990) & (df[COL["year"]] <= 2025)]

    return df

df = load_data()
YEARS = sorted(df[COL["year"]].unique().tolist())
LATEST = int(max(YEARS)) if YEARS else 2013

WORLD = alt.topo_feature(vega_data.world_110m.url, "countries")

def fmt_int(x):
    try: return f"{int(round(float(x))):,}".replace(",", " ")
    except: return "NA"

# =========================================================
# UI: 1 seul écran Carte + Zoom pays
# =========================================================
st.title("Fardeau mondial de la tuberculose — quantités absolues")

c1, c2 = st.columns([2,1])
with c1:
    year_sel = st.slider("Année", int(min(YEARS)), int(max(YEARS)), value=LATEST, step=1)
with c2:
    metric_alias = st.selectbox(
        "Métrique",
        options=["Prévalence (personnes)","Incidence (cas)","Décès (personnes)","Incidence TB/VIH (personnes)"]
    )

# Mapping métrique -> colonnes absolues et couleur
if metric_alias.startswith("Prévalence"):
    m_col, lo_col, hi_col, m_color = COL["prev_abs"], COL["prev_abs_lo"], COL["prev_abs_hi"], PALETTE["prev"]
elif metric_alias.startswith("Décès"):
    m_col, lo_col, hi_col, m_color = COL["deaths_abs"], COL["deaths_abs_lo"], COL["deaths_abs_hi"], PALETTE["mort"]
elif metric_alias.startswith("Incidence TB/VIH"):
    m_col, lo_col, hi_col, m_color = COL["tbhiv_abs"], COL["tbhiv_abs_lo"], COL["tbhiv_abs_hi"], "#8E44AD"
else:
    m_col, lo_col, hi_col, m_color = COL["inc_abs"], COL["inc_abs_lo"], COL["inc_abs_hi"], PALETTE["inc"]

# Sous-ensemble année + centroïdes
show = df[df[COL["year"]] == year_sel].copy()
show["country_norm"] = show[COL["country"]].apply(normalize_country_name)
show["lat"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[0])
show["lon"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[1])
show = show.dropna(subset=[m_col, "lat", "lon"])

# Échelle de taille: quantiles robustes
if not show.empty:
    q10 = np.nanpercentile(show[m_col], 10)
    q99 = np.nanpercentile(show[m_col], 99)
else:
    q10, q99 = 1, 10
size_scale = alt.Scale(domain=[q10, q99], range=[30, 1800])

# Fond gris
base_map = alt.Chart(WORLD).mark_geoshape(
    fill="#EEEEEE", stroke="white", strokeWidth=0.25
).project(type="equirectangular").properties(height=520)

# Bulles absolues
bubbles = alt.Chart(show).mark_circle(opacity=0.65, stroke="white", strokeWidth=0.5, color=m_color).encode(
    longitude="lon:Q", latitude="lat:Q",
    size=alt.Size(f"{m_col}:Q", title=metric_alias, scale=size_scale, legend=alt.Legend(orient="right")),
    tooltip=[
        alt.Tooltip(COL["country"], title="Pays"),
        alt.Tooltip(m_col, title=metric_alias, format=","),
        alt.Tooltip(lo_col, title="Borne basse", format=","),
        alt.Tooltip(hi_col, title="Borne haute", format=","),
    ]
).project(type="equirectangular").properties(height=520)

st.altair_chart(base_map + bubbles, use_container_width=True)

# Sélection pays + courbes absolues
st.markdown("### Zoom pays")
country = st.selectbox("Choisir un pays", options=sorted(df[COL["country"]].dropna().unique()))
dpc = df[df[COL["country"]] == country].sort_values(COL["year"]).copy()

if dpc.empty:
    st.info("Aucune donnée disponible pour ce pays.")
else:
    # Passage en format long sur les ABSOLUS uniquement
    long = dpc.rename(columns={
        COL["year"]: "Année",
        COL["inc_abs"]: "Incidence (cas)",
        COL["deaths_abs"]: "Décès (personnes)",
        COL["prev_abs"]: "Prévalence (personnes)",
        COL["tbhiv_abs"]: "Incidence TB/VIH (personnes)",
    })[["Année","Incidence (cas)","Décès (personnes)","Prévalence (personnes)","Incidence TB/VIH (personnes)"]]
    long = long.melt("Année", var_name="Métrique", value_name="Quantité")

    color_map = alt.Scale(
        domain=["Incidence (cas)","Décès (personnes)","Prévalence (personnes)","Incidence TB/VIH (personnes)"],
        range=[PALETTE["inc"], PALETTE["mort"], PALETTE["prev"], "#8E44AD"]
    )

    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Année:O"),
        y=alt.Y("Quantité:Q", title="Nombre de personnes / cas"),
        color=alt.Color("Métrique:N", scale=color_map),
        tooltip=["Année","Métrique", alt.Tooltip("Quantité:Q", format=",")]
    ).properties(height=340)

    st.altair_chart(line, use_container_width=True)

    # Tuiles dernière année
    last = dpc.iloc[-1]
    cA,cB,cC,cD = st.columns(4)
    cA.metric("Incidence (cas)", fmt_int(last[COL["inc_abs"]]))
    cB.metric("Décès (pers.)", fmt_int(last[COL["deaths_abs"]]))
    cC.metric("Prévalence (pers.)", fmt_int(last[COL["prev_abs"]]))
    cD.metric("TB/VIH (pers.)", fmt_int(last[COL["tbhiv_abs"]]))
