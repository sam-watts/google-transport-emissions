import streamlit as st
import numpy as np
import pandas as pd
import zipfile
import json
from tqdm import tqdm
from copy import deepcopy
import plotly.express as px
import reverse_geocoder as rg
from unicodedata import lookup
from google.oauth2 import service_account
from google.cloud import storage
import io


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


def extract_from_takeout(files: list[io.BytesIO]):
    data = []
    for file in files:
        with zipfile.ZipFile(file) as myzip:
            names = myzip.namelist()

            for name in tqdm(names):
                if name.startswith(
                    "Takeout/Location History/Semantic Location History"
                ):
                    year = int(name.split("/")[-2])
                    if year >= 2020:
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


def calculate_g_co2_eq(activity_type, distance_meters):
    match activity_type:
        case "CYCLING":
            # cycling average diet
            factor = 18.4
        case "RUNNING" | "WALKING":
            # assumed running is the same as walking (average diet)
            factor = 24.15
        case "IN_TRAIN":
            # mean of diesel and electric trains
            factor = (23.46 + 48.64) / 2
        case "IN_BUS":
            factor = 41.63
        case "FLYING":
            # mean of short and long haul flights
            factor = (239 + 340) / 2
        case "IN_PASSENGER_VEHICLE":
            # car (diesel) - no sharing!
            factor = 182.94
        case "IN_SUBWAY":
            # train (electric)
            factor = 23.46
        case "IN_TRAM":
            # train (electric)
            factor = 23.46
        case "UNKNOWN_ACTIVITY_TYPE":
            # TODO this is made up right now!
            factor = 100
        case _:
            factor = np.nan

    return distance_meters / 1000 * factor


def apply_rg_search(lat_series, lon_series):
    return [x["cc"] for x in rg.search(list(zip(lat_series, lon_series)))]


st.title("Google Timeline Carbon Footprint")
uploader_disabled = False

with st.sidebar:
    st.markdown("## Input Data")
    placeholder = st.empty()

    st.markdown("Or:")
    use_sample_data = st.checkbox("Use sample data?")

    if use_sample_data:
        uploader_disabled = True

    files_uploaded = placeholder.file_uploader(
        "Upload your Google Takeout .zip file:",
        type="zip",
        disabled=uploader_disabled,
        accept_multiple_files=True,
    )

    with st.expander(":bulb: How do I get my Google location data?"):
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

if use_sample_data:
    files = get_default_data()
else:
    files = files_uploaded

if not files:
    st.markdown("Select data in the sidebar to get started!")
else:
    reduction_levels = st.checkbox("Show emissions reduction levels?", value=True)
    raw_data = extract_from_takeout(files)

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

    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])
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

    df["travel_annotation"] = df.apply(
        lambda x: get_travel_annotation(x["start_country"], x["end_country"]), axis=1
    )

    aggregators = dict(
        total_distance_km=pd.NamedAgg("distance_meters", lambda x: x.sum() / 1000),
        tonnes_co2_eq=pd.NamedAgg("tonnes_co2_eq", "sum"),
        number_of_activities=pd.NamedAgg("activity_type", "count"),
    )

    agg = df.groupby(["year", "activity_type"]).agg(**aggregators).reset_index()

    tabs = st.tabs(
        [
            "Yearly Transport Emissions by Type",
            "Yearly Flight Emissions",
            "Global Emissions Reduction Targets to 2050",
        ]
    )

    with tabs[0]:
        fig = px.bar(
            agg,
            x="year",
            y="tonnes_co2_eq",
            color="activity_type",
        )
        fig.update_layout(yaxis_title="Tonnes CO2eq")
        fig.update_xaxes(type="category")
        if reduction_levels:
            fig.add_hline(
                y=3.5,
                line_dash="dash",
                annotation_text="3.5 tons CO2eq / person / year by 2030",
                line_color="red",
            )
            fig.add_hline(
                y=1.64,
                line_dash="dash",
                annotation_text="1.64 tons CO2eq / person / year by 2040",
                line_color="red",
            )
        st.plotly_chart(fig)

    with tabs[1]:
        fig = px.bar(
            df[df["activity_type"] == "FLYING"].sort_values("year"),
            x="year",
            y="tonnes_co2_eq",
            hover_data=[
                "distance_km",
                "duration_hours",
                "start_country",
                "end_country",
                "start_date",
            ],
            text="travel_annotation",
        )

        fig.update_xaxes(type="category")
        fig.update_layout(yaxis_title="Tonnes CO2eq")

        if reduction_levels:
            fig.add_hline(
                y=3.5,
                line_dash="dash",
                annotation_text="3.5 tons CO2eq / person / year by 2030",
                line_color="red",
            )
            fig.add_hline(
                y=1.64,
                line_dash="dash",
                annotation_text="1.64 tons CO2eq / person / year by 2040",
                line_color="red",
            )

        st.plotly_chart(fig)

    global_emissions_2022 = 40.6e9  # tonnes CO2eq
    years_to_2050 = np.linspace(2050, 2023, 28)
    emissions_per_year_zero_2050 = np.linspace(0, global_emissions_2022, 28)
    global_population = np.linspace(9.8e9, 8e9, 28)

    emissions_reductions = pd.DataFrame(
        dict(
            year=years_to_2050.astype(str),
            emissions_per_year_zero_2050=emissions_per_year_zero_2050,
            global_population=global_population,
            emissions_per_year_per_person_zero_2050=emissions_per_year_zero_2050
            / global_population,
        )
    ).loc[::-1]

    with tabs[2]:
        st.plotly_chart(
            px.line(
                emissions_reductions,
                x="year",
                y=[
                    "emissions_per_year_zero_2050",
                    "emissions_per_year_per_person_zero_2050",
                    "global_population",
                ],
            )
        )
