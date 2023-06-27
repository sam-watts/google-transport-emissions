import streamlit as st
import numpy as np
import pandas as pd
import zipfile
import json
from tqdm import tqdm
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
import reverse_geocoder as rg
from unicodedata import lookup
from google.oauth2 import service_account
from google.cloud import storage
import io
from datetime import date, datetime
from itertools import product
import inspect


@st.cache_data
def get_default_data():
    bucket_name = "takeout-transport-emissions"
    file_path = "takeout-20230301T101531Z-001.zip"
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    return [io.BytesIO(bucket.blob(file_path).download_as_string())]


def get_flag_emoji(country_iso_code: str):
    if len(country_iso_code) != 2:
        raise ValueError("Country ISO code must be 2 characters long")

    country_iso_code = country_iso_code.lower()
    return lookup(f"REGIONAL INDICATOR SYMBOL LETTER {country_iso_code[0]}") + lookup(
        f"REGIONAL INDICATOR SYMBOL LETTER {country_iso_code[1]}"
    )


def get_travel_annotation(from_cc, to_cc):
    return get_flag_emoji(from_cc) + " ➡ " + get_flag_emoji(to_cc)


def extract_from_takeout(files: list[io.BytesIO], start_date: date, end_date: date):
    data = []
    for file in files:
        with zipfile.ZipFile(file) as myzip:
            names = myzip.namelist()

            for name in tqdm(names):
                if name.startswith(
                    "Takeout/Location History/Semantic Location History"
                ):
                    year = int(name.split("/")[-2])
                    month_name = name.split("/")[-1].split("_")[-1].split(".")[0]
                    month = datetime.strptime(month_name, "%B").month

                    if start_date <= date(year, month, 1) <= end_date:
                        with myzip.open(name) as f:
                            data.append(json.load(f))

    print(f"Loaded {len(data)} months of location data")

    return data


def get_compound_key(activity, keys):
    if len(keys) > 0 and activity is not None:
        key = keys.pop(0)
        return get_compound_key(activity.get(key, None), keys)
    else:
        return activity


def parse_activities(data):
    # parse the data into a list of activities
    required_keys = [
        ["activityType"],
        ["distance"],
        ["confidence"],
        ["duration", "startTimestamp"],
        ["duration", "endTimestamp"],
        ["startLocation", "latitudeE7"],
        ["startLocation", "longitudeE7"],
        ["endLocation", "latitudeE7"],
        ["endLocation", "longitudeE7"],
    ]
    activities = []
    for d in data:
        for activity in d["timelineObjects"]:
            if (
                "activitySegment" in activity
                and "activityType" in activity["activitySegment"]
            ):
                activities.append(
                    {
                        ".".join(k): get_compound_key(activity["activitySegment"], k)
                        for k in deepcopy(required_keys)
                    }
                )

    return activities


# TODO - use confidence scores in some way:
# weighting of emissions?
# ignore low confidence activities completely?
def calculate_g_co2_eq(activity_type: str, distance_meters: float):
    # units: g CO2eq / person / km
    factors = {
        "CYCLING": 18.4,  # cycling average diet
        "RUNNING": 24.15,  # assumed running is the same as walking
        "WALKING": 24.15,
        "IN_TRAIN": (23.46 + 48.64) / 2,  # mean of diesel and electric trains
        "IN_BUS": 41.63,
        "FLYING": (239 + 340) / 2,  # mean of short and long haul flights
        "IN_PASSENGER_VEHICLE": 182.94,  # car (diesel) - no sharing!
        "IN_SUBWAY": 23.46,  # train (electric)
        "IN_TRAM": 23.46,  # train (electric)
    }

    if activity_type in factors:
        factor = factors[activity_type]
    elif activity_type == "UNKNOWN_ACTIVITY_TYPE":
        # when google can't figure out what you're doing, use the mean
        factor = np.mean(list(factors.values()))
    else:
        # ignore all other activity types
        factor = np.nan

    return distance_meters / 1000 * factor


def apply_rg_search(lat_series, lon_series):
    return [x["cc"] for x in rg.search(list(zip(lat_series, lon_series)))]


st.set_page_config(
    page_title="Transport CO2 Emissions",
    page_icon="🌍️",
    initial_sidebar_state="expanded",
)

st.title("Google Timeline Carbon Footprint")
uploader_disabled = False

with st.sidebar:
    st.markdown("## Input Data")
    placeholder = st.empty()

    with st.expander(":bulb: How do I get my Google Timeline data?"):
        st.markdown(
            """
            1. Go to [takeout.google.com](https://takeout.google.com/) - make
               sure you are signed in with the correct google account if you
               have more than one
            2. Select "Deselect all" at the top
            3. Scroll down and select "Location History"
            4. Click "Next step"
            5. Click "Create export"
            6. Wait for the email to arrive with the download link
            7. Upload your zip file above
            """
        )

    st.markdown("**Or:**")
    use_sample_data = st.checkbox("Use sample data?")

    if use_sample_data:
        uploader_disabled = True

    files_uploaded = placeholder.file_uploader(
        "Upload your Google Timeline .zip file:",
        type="zip",
        disabled=uploader_disabled,
        accept_multiple_files=True,
    )

    start_date, end_date = st.slider(
        "Select a date range:",
        date(2017, 1, 1),
        date.today(),
        (date(2019, 1, 1), date.today()),
    )


if use_sample_data:
    files = get_default_data()
else:
    files = files_uploaded

if not files:
    st.markdown(
        """
        👋 Welcome! This app lets you see the approximate carbon footprint
            of your travel, based on your Google location data.
            This works best if you have an Android phone, and have had
            location history enabled for at least a few months.

        :arrow_left::floppy_disk: **Add a data source in the sidebar to get
            started!**

        Put simply - if you fly a lot, this may be far outweighing
        other efforts you might be
        making to reduce your impact on the climate:
        """
    )
    with st.columns([0.15, 0.7, 0.15])[1]:
        st.image("flight_free_uk.jpg", caption="Source: https://flightfree.co.uk/")
    st.markdown(
        """
        You can find my blog post discussing wider issues around aviation [here](
            https://sam-watts.github.io/blog/posts/google-timeline-emissions/
        )

        This is not intended to be a precise measurement of your emissions, but
        an indication of the order of magnitude compared to the average person
        in different countries. Google is not always able to
        accurately determine the mode of transport you used.

        If you'd like to verify yourself that this app is not doing anything
        sketchy with your data (or if you're just curious), you can check out
        the source code on
        [GitHub](https://github.com/sam-watts/google_transport_emissions)
        """
    )
else:
    reduction_levels = st.checkbox(
        "Show 1.5°C emissions reduction targets?",
        value=False,
        help=(
            "These dotted red lines show the required CO2eq emissions / person / year"
            " up to 2050, to have a 50% chance of limiting global warming to 1.5°C."
            " This includes all emissions, not just transport. See"
            " the 'Global Emissions Reduction Targets to 2050' tab for the numbers and"
            " logic this is based on."
        ),
    )
    raw_data = extract_from_takeout(files, start_date, end_date)

    parsed_data = parse_activities(raw_data)
    df = pd.DataFrame(parsed_data)

    df = df.rename(
        columns={
            "activityType": "activity_type",
            "distance": "distance_meters",
            "duration.startTimestamp": "start_timestamp",
            "duration.endTimestamp": "end_timestamp",
        }
    )

    # convert to regular lat/lon units
    df["start_location_latitude"] = df["startLocation.latitudeE7"] / 10_000_000
    df["start_location_longitude"] = df["startLocation.longitudeE7"] / 10_000_000
    df["end_location_latitude"] = df["endLocation.latitudeE7"] / 10_000_000
    df["end_location_longitude"] = df["endLocation.longitudeE7"] / 10_000_000

    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], format="ISO8601")
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], format="ISO8601")
    df["start_date"] = df["start_timestamp"].dt.date
    df["end_date"] = df["end_timestamp"].dt.date
    df["duration_hours"] = round(
        (df["end_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 3600, 2
    )

    df["start_country"] = apply_rg_search(
        df["start_location_latitude"], df["start_location_longitude"]
    )
    df["end_country"] = apply_rg_search(
        df["end_location_latitude"], df["end_location_longitude"]
    )

    df["year"] = df["start_timestamp"].dt.year
    df["month"] = df["start_timestamp"].dt.month

    df["distance_km"] = df["distance_meters"] / 1000
    df["tonnes_co2_eq"] = (
        df.apply(
            lambda x: calculate_g_co2_eq(x["activity_type"], x["distance_meters"]),
            axis=1,
        )
        / 1_000_000
    )

    df = df.dropna(subset=["tonnes_co2_eq"])
    df["travel_annotation"] = df.apply(
        lambda x: get_travel_annotation(x["start_country"], x["end_country"]), axis=1
    )

    df = df.sort_values("start_timestamp")

    aggregators = dict(
        total_distance_km=pd.NamedAgg("distance_meters", lambda x: x.sum() / 1000),
        tonnes_co2_eq=pd.NamedAgg("tonnes_co2_eq", "sum"),
        number_of_activities=pd.NamedAgg("activity_type", "count"),
    )

    agg = df.groupby(["year", "activity_type"]).agg(**aggregators).reset_index()

    agg["percentage_yearly_emissions"] = (
        (agg["tonnes_co2_eq"] / agg.groupby("year")["tonnes_co2_eq"].transform("sum"))
        * 100
    ).map("{:,.2f}%".format)

    tabs = st.tabs(
        [
            "Yearly Transport Emissions by Mode of Travel",
            "Yearly Flight Emissions",
            "Global Emissions Reduction Targets to 2050",
        ]
    )

    with tabs[0]:

        def clean_columns(x):
            return (
                x.str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
                .str.replace("-", "_")
                .str.replace("_+", "_", regex=True)
            )

        emissions_by_country = pd.read_csv("data/per-capita-co2-transport.csv")

        emissions_by_country.columns = clean_columns(emissions_by_country.columns)
        emissions_by_country = emissions_by_country.rename(
            columns={"transport_per_capita": "tonnes_co2_eq"}
        )
        del emissions_by_country["code"]

        fill_years = range(2020, date.today().year + 1)
        extra_rows = pd.DataFrame(
            product(emissions_by_country["entity"].unique(), fill_years),
            columns=["entity", "year"],
        )
        extra_rows["tonnes_co2_eq"] = np.nan

        emissions_by_country = pd.concat([emissions_by_country, extra_rows], axis=0)
        emissions_by_country = emissions_by_country.sort_values(["entity", "year"])
        emissions_by_country = emissions_by_country.fillna(method="ffill")

        emissions_by_country_international_flights = pd.read_csv(
            "data/per-capita-co2-international-flights-adjusted.csv"
        )
        emissions_by_country_international_flights.columns = clean_columns(
            emissions_by_country_international_flights.columns
        )

        emissions_by_country = pd.merge(
            emissions_by_country,
            emissions_by_country_international_flights[
                ["entity", "per_capita_international_co2_adjusted"]
            ],
            on="entity",
        )

        # add the international flights to the transport emissions,
        # dividing by 1000 to convert to tonnes
        emissions_by_country["tonnes_co2_eq"] += (
            emissions_by_country["per_capita_international_co2_adjusted"] / 1000
        )

        # add in estimated emissions from shipping, a flat 11.6% of total
        # emissions globally
        shipping_estimate = 0.116 * emissions_by_country["tonnes_co2_eq"] / (1 - 0.116)
        emissions_by_country["tonnes_co2_eq"] += shipping_estimate

        emissions_by_country = emissions_by_country[
            emissions_by_country["year"].isin(agg["year"].unique())
        ]
        comp_countries = st.multiselect(
            (
                "Select a country to compare your results to their yearly per capita"
                " transport emissions across all modes:"
            ),
            emissions_by_country["entity"].unique(),
        )

        fig = px.bar(
            agg,
            x="year",
            y="tonnes_co2_eq",
            color="activity_type",
            hover_data=[
                "percentage_yearly_emissions",
                "number_of_activities",
                "total_distance_km",
            ],
            labels={"tonnes_co2_eq": "Tonnes CO2eq", "activity_type": "Mode of Travel"},
        )
        # fig.update_layout(yaxis_title="Tonnes CO2eq")
        fig.update_xaxes(type="category")
        if reduction_levels:
            fig.add_hline(
                y=3.5,
                line_dash="dash",
                annotation_text="3.5 tonnes CO2eq / person / year by 2030",
                line_color="red",
            )
            fig.add_hline(
                y=1.64,
                line_dash="dash",
                annotation_text="1.64 tonnes CO2eq / person / year by 2040",
                line_color="red",
            )

        for country in comp_countries:
            fig.add_trace(
                go.Scatter(
                    x=emissions_by_country[emissions_by_country["entity"] == country][
                        "year"
                    ],
                    y=emissions_by_country[emissions_by_country["entity"] == country][
                        "tonnes_co2_eq"
                    ],
                    mode="lines",
                    name=f"{country} per capita transport emissions",
                    line=dict(dash="dash"),
                )
            )

        st.plotly_chart(fig)

        with st.expander(":test_tube: Sources"):
            st.markdown(
                """
                ### Transport Emissions Factors

                These are taken from Oliver Corradi's great general
                [post](https://oliviercorradi.com/climate-change/#transportation--reduce-long-distance-travels)
                on climate change. The code to calculate the emissions is show below:
                """
            )
            st.markdown(f"```python \n{inspect.getsource(calculate_g_co2_eq)}```")
            st.markdown(
                """
                ### Per Capita Transport Emissions by Country

                Several sources from Our World in Data Were used, as there is no
                single source I could find for this data across all transport
                sectors at the time of writing (please reach out if I'm wrong
                here!)

                Drawbacks:
                * Doesn't account for non-CO2 greenhouse gas emissions
                * Doesn't accurately account for shipping emissions - a global
                adjustment factor is used due to lack of data
                * Data age - see indiviual points below

                [per-capita-co2-transport.csv - up to 2019](
                    https://ourworldindata.org/grapher/per-capita-co2-transport):

                > "... the average per capita emissions of carbon dioxide from
                > transport each year. This includes road, train, bus and
                > domestic air travel but does not include international aviation
                > and shipping."

                Data from 2019 is used to fill in missing data for 2020 onwards.

                [per-capita-co2-international-flights-adjusted.csv - 2018 only](
                    https://ourworldindata.org/grapher/per-capita-co2-international-flights-adjusted):

                > "International aviation emissions are allocated to the country of
                > departure, then adjusted for tourism by multiplying this figure
                > by the ratio of inbound-to-outbound travelers. This attempts to
                > distinguish between locals traveling abroad and foreign visitors
                > traveling to that country on the same flight."

                As the data only covers one year, it is used to fill in all other years.

                [Global transport totals](
                    https://ourworldindata.org/transport#co2-emissions-by-mode-of-transport)

                This data is used to correct for the fact that the above data
                does not account for shipping. A global adjustment factor of
                10.6% is used.

                """
            )

    with tabs[1]:
        flight_emissions_by_country = pd.read_csv(
            "data/per-capita-co2-aviation-adjusted.csv"
        )
        flight_emissions_by_country.columns = clean_columns(
            flight_emissions_by_country.columns
        )

        flight_emissions_by_country = flight_emissions_by_country.rename(
            columns={"per_capita_aviation_co2_adjusted": "tonnes_co2_eq"}
        )

        comp_countries_flights = st.multiselect(
            (
                "Select a country to compare your results to their yearly per capita"
                " transport emissions from flying:"
            ),
            flight_emissions_by_country["entity"].unique(),
        )

        user_flights_only = df[df["activity_type"] == "FLYING"]

        fig = px.bar(
            user_flights_only,
            x="year",
            y="tonnes_co2_eq",
            hover_data=[
                "distance_km",
                "duration_hours",
                "start_country",
                "end_country",
                "start_date",
            ],
            barmode="stack",
            text="travel_annotation",
        )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

        fig.update_xaxes(type="category")
        fig.update_layout(yaxis_title="Tonnes CO2eq")

        fig.update_layout(margin=dict(r=100))
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=1.05,
            y=0.2,
            showarrow=False,
            text="January",
            xanchor="left",
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=1.05,
            y=0.7,
            showarrow=False,
            text="December",
            xanchor="left",
        )
        # add an arrow pointing from the january annotation to the december
        # annotation
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            ayref="y domain",
            ay=0.15,
            y=0.75,
            x=1.03,
            ax=1.03,
            showarrow=True,
            arrowcolor="white",
            arrowhead=4,
            align="center",
        )

        if reduction_levels:
            fig.add_hline(
                y=3.5,
                line_dash="dash",
                annotation_text="3.5 tonnes CO2eq / person / year by 2030",
                line_color="red",
            )
            fig.add_hline(
                y=1.64,
                line_dash="dash",
                annotation_text="1.64 tonnes CO2eq / person / year by 2040",
                line_color="red",
            )

        for country in comp_countries_flights:
            fig.add_trace(
                go.Scatter(
                    x=user_flights_only["year"].unique(),
                    y=[
                        flight_emissions_by_country[
                            flight_emissions_by_country["entity"] == country
                        ]["tonnes_co2_eq"].values[0]
                        / 1000
                    ]
                    * user_flights_only["year"].unique().shape[0],
                    mode="lines",
                    name=f"{country} per capita aviation emissions",
                    line=dict(dash="dash"),
                )
            )

        st.plotly_chart(fig)

        with st.expander(":test_tube: Sources"):
            st.markdown(
                """
                [per-capita-co2-avaiation-adjusted.csv - 2018 only](
                    https://ourworldindata.org/grapher/per-capita-co2-aviation-adjusted
                )
                """
            )

    global_emissions_2022 = 40.6e9  # tonnes CO2eq
    years_to_2050 = np.linspace(2050, 2023, 28)
    emissions_per_year_zero_2050 = np.linspace(0, global_emissions_2022, 28)
    global_population = np.linspace(9.8e9, 8e9, 28)

    emissions_reductions = pd.DataFrame(
        dict(
            year=years_to_2050.astype(str),
            emissions_per_year_tonnes_co2eq=emissions_per_year_zero_2050,
            global_population=global_population,
            emissions_per_year_per_person_tonnes_co2eq=emissions_per_year_zero_2050
            / global_population,
        )
    ).loc[::-1]

    emissions_reductions = emissions_reductions.melt(
        id_vars="year",
        value_vars=[
            "emissions_per_year_tonnes_co2eq",
            "emissions_per_year_per_person_tonnes_co2eq",
            "global_population",
        ],
    )

    with tabs[2]:
        st.markdown(
            """
            According to the Global Carbon Project's [2022 estimates](
            https://globalcarbonbudget.org/wp-content/uploads/Key-messages.pdf),
            global emissions in 2022 reached 40.6 GtCO2.

            > "Reaching zero CO2 emissions by 2050 entails a total
            > anthropogenic CO2 emissions linear
            > decrease by about 1.4 GtCO2 each year, comparable to the observed
            > fall in 2020 emissions
            > resulting from COVID-19 measures, highlighting the scale of the
            > action needed. This would
            > result in additional emissions of 560 GtCO 2 from year 2023,
            > leading to a 50% likelihood to
            > limit warming around 1.6°C"

            The graphs below show the emissions per year and per person required
            to reach zero CO2 emissions by 2050. Population is linearly
            projected to 9.8 billion in 2050.
            """
        )

        st.plotly_chart(
            px.line(
                emissions_reductions,
                x="year",
                y="value",
                facet_row="variable",
                facet_row_spacing=0.10,
            )
            .update_layout(margin=dict(r=250))
            .update_yaxes(matches=None)
            .for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=")[-1].replace("_", " ").title(), textangle=0
                )
            )
        )

st.info(
    (
        "If you have any questions or feedback, please create an issue"
        " [here](https://github.com/sam-watts/google-transport-emissions/issues/new)"
    ),
    icon="ℹ️",
)
