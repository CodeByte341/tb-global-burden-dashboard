import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from vega_datasets import data as vega_data

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Global Tuberculosis Burden — Analytical Dashboard",
    layout="wide"
)
alt.themes.enable("opaque")

PALETTE = {
    "inc":  "#2E86DE",
    "mort": "#E74C3C",
    "prev": "#F39C12",
    "neutral": "#7F8C8D",
}
# Region color palette (legend label = "Region")
REGION_COLORS = {
    "AFR": "#9B59B6",  # Africa
    "AMR": "#27AE60",  # Americas
    "EMR": "#E67E22",  # Eastern Mediterranean
    "EUR": "#2980B9",  # Europe
    "SEA": "#C0392B",  # South-East Asia
    "WPR": "#16A085",  # Western Pacific
}

REGION_FULL = {
    "AFR": "Africa",
    "AMR": "Americas",
    "EMR": "Eastern Mediterranean",
    "EUR": "Europe",
    "SEA": "South-East Asia",
    "WPR": "Western Pacific"
}

# =========================================================
# CSV COLUMN MAP (exact names, file never modified)
# =========================================================
COL = {
    "country": "Country or territory name",
    "iso3": "ISO 3-character country/territory code",
    "region": "Region",
    "year": "Year",
    "pop": "Estimated total population number",

    # Absolute quantities
    "prev_abs":   "Estimated prevalence of TB (all forms)",
    "prev_abs_lo":"Estimated prevalence of TB (all forms), low bound",
    "prev_abs_hi":"Estimated prevalence of TB (all forms), high bound",

    "deaths_abs":   "Estimated number of deaths from TB (all forms, excluding HIV)",
    "deaths_abs_lo":"Estimated number of deaths from TB (all forms, excluding HIV), low bound",
    "deaths_abs_hi":"Estimated number of deaths from TB (all forms, excluding HIV), high bound",

    "inc_abs":   "Estimated number of incident cases (all forms)",
    "inc_abs_lo":"Estimated number of incident cases (all forms), low bound",
    "inc_abs_hi":"Estimated number of incident cases (all forms), high bound",

    # Per-100k (for map options)
    "prev_100k": "Estimated prevalence of TB (all forms) per 100 000 population",
    "mort_100k": "Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population",
    "inc_100k":  "Estimated incidence (all forms) per 100 000 population",

    # Optional tiles (not on map)
    "cdr_pct": "Case detection rate (all forms), percent",
}

# =========================================================
# COUNTRY NAME NORMALIZATION (for centroid matching)
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
# COUNTRY CENTROIDS (lat, lon) — main coverage
# =========================================================
COUNTRY_CENTROIDS = {
    # Americas
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
    # Africa
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
    # Middle East / Central Asia
    "Israel":(31.20,34.86), "Lebanon":(33.92,35.89), "Syria":(34.80,38.98),
    "Jordan":(31.24,36.76), "Iraq":(33.22,43.68), "Saudi Arabia":(23.94,45.08),
    "Yemen":(15.55,48.52), "Oman":(20.59,56.09), "United Arab Emirates":(24.23,53.66),
    "Qatar":(25.30,51.15), "Bahrain":(26.07,50.55), "Kuwait":(29.27,47.50),
    "Iran":(32.43,53.69), "Afghanistan":(33.94,67.71), "Pakistan":(29.95,69.35),
    "Azerbaijan":(40.35,47.70), "Armenia":(40.29,44.94), "Georgia":(42.32,43.37),
    "Kazakhstan":(48.16,67.30), "Uzbekistan":(41.38,64.57), "Turkmenistan":(39.10,59.37),
    "Kyrgyzstan":(41.46,74.56), "Tajikistan":(38.86,71.27),
    # South & East Asia
    "India":(22.88,79.80), "Sri Lanka":(7.86,80.68), "Nepal":(28.39,84.12),
    "Bhutan":(27.41,90.43), "Bangladesh":(23.69,90.35), "Maldives":(3.67,73.54),
    "Myanmar":(19.75,96.10), "Thailand":(15.12,101.00), "Laos":(19.86,102.50),
    "Cambodia":(12.69,104.90), "Vietnam":(15.62,106.25), "Malaysia":(4.21,109.20),
    "Singapore":(1.35,103.82), "Indonesia":(-2.60,118.02), "Philippines":(12.75,122.73),
    "Brunei":(4.52,114.72), "East Timor":(-8.79,125.85),
    "China":(35.86,104.19), "Mongolia":(46.86,103.84), "Japan":(36.20,138.25),
    "South Korea":(36.50,127.98), "North Korea":(40.34,127.51), "Taiwan":(23.70,121.08),
    # Oceania
    "Australia":(-25.27,133.77), "New Zealand":(-41.29,174.78),
    "Papua New Guinea":(-6.31,146.38), "Fiji":(-17.82,178.13), "Solomon Islands":(-9.23,160.14),
    "Vanuatu":(-15.38,166.96), "Samoa":(-13.76,-172.10), "Tonga":(-21.18,-175.20),
    "Micronesia":(6.88,158.22), "Palau":(7.50,134.62), "Kiribati":(1.87,-157.36),
    "Marshall Islands":(7.12,171.18), "Nauru":(-0.53,166.93), "Tuvalu":(-7.11,177.65),
}

# =========================================================
# LOAD DATA (root or data/), compute cumulative
# =========================================================
@st.cache_data(show_spinner=True)
def load_data():
    candidates = ["TB_Burden_Country.csv", os.path.join("data", "TB_Burden_Country.csv")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("CSV not found. Place it at repository root or in data/.")
        st.stop()

    raw = pd.read_csv(path)

    needed = list(COL.values())
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

    df = raw[needed].copy()

    # numeric types
    for c in [COL["year"], COL["pop"],
              COL["prev_abs"], COL["prev_abs_lo"], COL["prev_abs_hi"],
              COL["deaths_abs"], COL["deaths_abs_lo"], COL["deaths_abs_hi"],
              COL["inc_abs"], COL["inc_abs_lo"], COL["inc_abs_hi"],
              COL["prev_100k"], COL["mort_100k"], COL["inc_100k"],
              COL["cdr_pct"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # helpers
    df["country_norm"] = df[COL["country"]].apply(normalize_country_name)
    df["region_key"] = df[COL["region"]].astype(str).str.upper().str[:3]
    df["region_key"] = df["region_key"].where(df["region_key"].isin(REGION_COLORS.keys()), "Other")

    # cumulative deaths per country
    df = df.sort_values([COL["country"], COL["year"]]).copy()
    df["cum_deaths_abs"] = df.groupby(COL["country"])[COL["deaths_abs"]].cumsum()

    # global aggregates per year
    global_year = (
        df.groupby(COL["year"])
          .agg(inc_abs=(COL["inc_abs"], "sum"),
               deaths_abs=(COL["deaths_abs"], "sum"),
               prev_abs=(COL["prev_abs"], "sum"))
          .reset_index()
          .rename(columns={COL["year"]:"year"})
    )
    global_year["cum_deaths_abs"] = global_year["deaths_abs"].cumsum()

    return df, global_year

df, global_year = load_data()
YEARS = sorted(df[COL["year"]].dropna().unique().tolist())
EARLY = int(min(YEARS)) if YEARS else None
LATEST = int(max(YEARS)) if YEARS else None

WORLD = alt.topo_feature(vega_data.world_110m.url, "countries")

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return "NA"

# =========================================================
# HEADER — Title / Subtitle / Authors
# =========================================================
st.title("Global Tuberculosis Burden — Analytical Dashboard")
st.markdown(
    "<div style='color:#555;font-size:16px;margin-top:-6px;'>"
    "Data Mining & Data Visualization coursework — Team: "
    "<b>Axel Fontanier</b>, <b>Romain Gille</b>, <b>Solenn Lorient</b>, <b>Flavie Perrier</b>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
"""
**Context.** Tuberculosis (TB) is an infectious disease caused by *Mycobacterium tuberculosis*. It mainly affects the lungs, spreads via airborne droplets, and can be fatal without timely treatment.

**Detection and treatment.** Diagnosis combines clinical assessment with microbiological confirmation (sputum smear microscopy, rapid molecular tests such as Xpert, and culture where available), complemented by chest radiography when indicated. Standard multi-drug regimens cure drug-susceptible TB; drug-resistant TB requires longer, specialized treatment. Early case-finding and full treatment completion are essential to reduce transmission and deaths.

**What each metric means (and when to use absolute vs per-100k):**
- **Incidence**: the number of **new and relapse TB cases** occurring in a year.  
  • *Absolute* shows the **total workload** for health systems.  
  • *Per-100k* normalizes by population, enabling **fair comparisons** between countries of different sizes.
- **Deaths**: the **number of people who died** from TB during the year (as reported in the dataset).  
  • *Absolute* reflects the **total fatal toll**.  
  • *Per-100k* highlights **mortality risk** relative to population size.
- **Prevalence**: the **total number of people living with active TB disease** during the year.  
  • *Absolute* indicates the **clinical caseload** at a point or period in time.  
  • *Per-100k* helps compare **disease burden intensity** across populations.

**How to use this dashboard.**  
- Explore the **world bubble map**: bubble **size** equals the selected metric (absolute counts or per-100k); bubble **color** encodes **Region**.  
- Switch between **Global view** and **Country zoom**. Both views include **cumulative deaths** to track the long-term fatal burden.
"""
)

# =========================================================
# VIEW SELECTOR + MAP CONTROLS
# =========================================================
view = st.selectbox("View", options=["Global view", "Country zoom"])

if view == "Global view":
    c1, c2, c3 = st.columns([2, 1.6, 1])
else:
    c1, c2, c3 = st.columns([2, 1.6, 1.6])

with c1:
    year_sel = st.slider("Year", int(min(YEARS)), int(max(YEARS)), value=LATEST, step=1)

with c2:
    metric_choice = st.selectbox(
        "Map metric",
        options=[
            "Incidence (absolute)",
            "Deaths (absolute)",
            "Prevalence (absolute)",
            "Incidence per 100k",
            "Deaths per 100k",
            "Prevalence per 100k",
        ],
        index=0
    )

with c3:
    if view == "Country zoom":
        country_sel = st.selectbox("Country", options=sorted(df[COL["country"]].dropna().unique()))
    else:
        st.markdown(
            "<div style='margin-top:28px;color:#7f8c8d'>Legend: size = metric, color = Region</div>",
            unsafe_allow_html=True
        )

# Map metric mapping
if metric_choice == "Incidence (absolute)":
    m_col, label = COL["inc_abs"], "Incidence (absolute)"
elif metric_choice == "Deaths (absolute)":
    m_col, label = COL["deaths_abs"], "Deaths (absolute)"
elif metric_choice == "Prevalence (absolute)":
    m_col, label = COL["prev_abs"], "Prevalence (absolute)"
elif metric_choice == "Incidence per 100k":
    m_col, label = COL["inc_100k"], "Incidence per 100k"
elif metric_choice == "Deaths per 100k":
    m_col, label = COL["mort_100k"], "Deaths per 100k"
else:
    m_col, label = COL["prev_100k"], "Prevalence per 100k"

# Data for the selected year
show = df[df[COL["year"]] == year_sel].copy()
show["country_norm"] = show[COL["country"]].apply(normalize_country_name)
show["lat"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[0])
show["lon"] = show["country_norm"].map(lambda x: COUNTRY_CENTROIDS.get(x, (np.nan, np.nan))[1])
show = show.dropna(subset=[m_col, "lat", "lon"]).copy()

# If country zoom, keep only the selected country on the map
if view == "Country zoom":
    show = show[show[COL["country"]] == country_sel]

# Region coloring
region_domain = list(REGION_COLORS.keys())
region_range = [REGION_COLORS[k] for k in region_domain]

# Size scaling
if show.empty:
    q10, q99 = 1, 10
else:
    q10 = np.nanpercentile(show[m_col], 10)
    q99 = np.nanpercentile(show[m_col], 99)
    if q10 == q99:
        q10, q99 = 0, max(q99, 1)
size_scale = alt.Scale(domain=[q10, q99], range=[30, 1800])

# Background map
base_map = alt.Chart(alt.topo_feature(vega_data.world_110m.url, "countries")).mark_geoshape(
    fill="#EEEEEE", stroke="white", strokeWidth=0.3
).project(type="equirectangular").properties(height=520)

# Bubbles
bubbles = alt.Chart(show).mark_circle(opacity=0.8, stroke="white", strokeWidth=0.6).encode(
    longitude="lon:Q",
    latitude="lat:Q",
    size=alt.Size(f"{m_col}:Q", title=label, scale=size_scale),
    color=alt.Color("region_key:N", title="Region", scale=alt.Scale(domain=region_domain, range=region_range)),
    tooltip=[
        alt.Tooltip(COL["country"], title="Country"),
        alt.Tooltip(COL["region"], title="Region"),
        alt.Tooltip(COL["pop"], title="Population", format=","),
        alt.Tooltip(m_col, title=label, format=",.0f" if "absolute" in label else ".1f"),
        alt.Tooltip(COL["deaths_abs"], title="Deaths (absolute)", format=","),
        alt.Tooltip("cum_deaths_abs:Q", title="Cumulative deaths to year", format=",")
    ]
).project(type="equirectangular").properties(height=520)

st.altair_chart(base_map + bubbles, use_container_width=True)

# =========================================================
# VIEWS BELOW THE MAP
# =========================================================
if view == "Global view":
    st.markdown("### Global time series and cumulative deaths")
    gy = global_year.copy()

    lines_df = gy.melt("year", value_vars=["inc_abs","deaths_abs","prev_abs"],
                       var_name="Metric", value_name="Value").replace({
        "inc_abs":"Incidence (absolute)",
        "deaths_abs":"Deaths (absolute)",
        "prev_abs":"Prevalence (absolute)"
    })
    color_map = alt.Scale(domain=["Incidence (absolute)","Deaths (absolute)","Prevalence (absolute)"],
                          range=[PALETTE["inc"], PALETTE["mort"], PALETTE["prev"]])

    ts = alt.Chart(lines_df).mark_line(point=True).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("Value:Q", title="People / cases"),
        color=alt.Color("Metric:N", scale=color_map),
        tooltip=["year","Metric", alt.Tooltip("Value:Q", format=",")]
    ).properties(height=320)

    cum = alt.Chart(gy).mark_area(opacity=0.2, color=PALETTE["mort"]).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("cum_deaths_abs:Q", title="Cumulative deaths (absolute)"),
        tooltip=["year", alt.Tooltip("cum_deaths_abs:Q", format=",")]
    ).properties(height=200)

    st.altair_chart(ts, use_container_width=True)
    st.altair_chart(cum, use_container_width=True)

else:
    st.markdown("### Country time series and cumulative deaths")
    dpc = df[df[COL["country"]] == country_sel].sort_values(COL["year"]).copy()

    long = dpc.rename(columns={
        COL["year"]:"Year",
        COL["inc_abs"]:"Incidence (absolute)",
        COL["deaths_abs"]:"Deaths (absolute)",
        COL["prev_abs"]:"Prevalence (absolute)",
    })[["Year","Incidence (absolute)","Deaths (absolute)","Prevalence (absolute)"]].melt(
        "Year", var_name="Metric", value_name="Value"
    )

    color_map = alt.Scale(domain=["Incidence (absolute)","Deaths (absolute)","Prevalence (absolute)"],
                          range=[PALETTE["inc"], PALETTE["mort"], PALETTE["prev"]])

    ts = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Year:O"),
        y=alt.Y("Value:Q", title="People / cases"),
        color=alt.Color("Metric:N", scale=color_map),
        tooltip=["Year","Metric", alt.Tooltip("Value:Q", format=",")]
    ).properties(height=340)

    dpc_cum = dpc[[COL["year"], COL["deaths_abs"]]].rename(columns={COL["year"]:"Year", COL["deaths_abs"]:"Deaths"})
    dpc_cum["Cumulative deaths"] = dpc_cum["Deaths"].cumsum()

    cum = alt.Chart(dpc_cum).mark_area(opacity=0.2, color=PALETTE["mort"]).encode(
        x=alt.X("Year:O"),
        y=alt.Y("Cumulative deaths:Q", title="Cumulative deaths (absolute)"),
        tooltip=["Year", alt.Tooltip("Cumulative deaths:Q", format=",")]
    ).properties(height=200)

    st.altair_chart(ts, use_container_width=True)
    st.altair_chart(cum, use_container_width=True)

    # Latest tiles
    last = dpc.iloc[-1]
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Incidence (absolute)", fmt_int(last[COL["inc_abs"]]))
    cB.metric("Deaths (absolute)", fmt_int(last[COL["deaths_abs"]]))
    cC.metric("Prevalence (absolute)", fmt_int(last[COL["prev_abs"]]))
    cD.metric("Case detection rate (%)", f"{last[COL['cdr_pct']]:.0f}" if not pd.isna(last[COL["cdr_pct"]]) else "NA")

# =========================================================
# AUTOMATED SYNTHESIS — Which regions are most affected?
# =========================================================
st.markdown("---")
st.markdown("### Overall synthesis")

st.markdown(
"""
**Where the burden concentrates.** In most years of global reporting, a **large share of the absolute TB burden** is concentrated in a limited number of countries across **South-East Asia** and parts of **Africa**, with comparatively **lower levels** observed in much of **Europe** and the **Americas**. This pattern reflects both **population size** and **underlying epidemiology**.

**Absolute vs per-100k perspectives.** Countries with very large populations can dominate **absolute** incidence and deaths even when their **per-100k** rates are moderate. Conversely, smaller countries may exhibit high **per-100k** rates without contributing as much to the global **absolute** totals. Using both lenses together prevents misleading conclusions.

**Why some places are less affected.** Lower TB burden typically aligns with a combination of factors:  
- **Earlier detection** and rapid **access to effective diagnostics** (including molecular tests),  
- Strong **treatment programs** with high completion rates and patient support,  
- **Socio-economic conditions** that reduce transmission risk (housing, nutrition, working conditions),  
- Robust **primary care** and **infection-control** standards, and  
- Childhood **BCG vaccination**, which helps protect against severe forms in children, even though it offers **limited protection** against pulmonary TB in adults.

**Why some places remain highly affected.** Persistent burden is often associated with **delayed diagnosis**, **treatment interruptions**, **health-system constraints**, **poverty and crowding**, **co-morbidities** (including HIV and diabetes), and the presence of **drug-resistant TB**. These drivers can sustain transmission and increase mortality despite progress elsewhere.

**Programmatic implication.** Sustainable gains come from pairing **population-level prevention** and **social protection** with **high-quality case-finding** and **treatment completion**. Monitoring **absolute counts** guides planning and logistics, while **per-100k** indicators support benchmarking, prioritization, and evaluation of equitable progress across regions and countries.
"""
)

# =========================================================
# SOURCES (single location)
# =========================================================
st.markdown("---")
st.markdown(
"""
**Sources.** World Health Organization (WHO) data.  
- WHO Data Portal: https://www.who.int/data/
- The Site Where We Recover The Dataset: https://www.tableau.com/learn/articles/free-public-data-sets
- The Associated Study: https://iris.who.int/server/api/core/bitstreams/07cb19ac-9dbe-483a-a7e1-b23505b0e081/content
"""
)
