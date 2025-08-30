import streamlit as st
import sqlite3
import pandas as pd
import altair as alt

"""
streamlit run monitor.py
"""

DB_FILE = "Monitoring_Tool/monitoring.db"

def load_data(table):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY timestamp DESC", conn)

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")

st.title("SmallGuardians Monitoring Dashboard")

# Tabs para separar vistas
tab1, tab2, tab3 = st.tabs(["Input Events", "Output Events", "Statistics"])

with tab1:
    st.subheader("Input Events")
    input_df = load_data("input_events")
    st.dataframe(input_df)

with tab2:
    st.subheader("Output Events")
    output_df = load_data("output_events")
    st.dataframe(output_df)

with tab3:
    st.subheader("Statistics")

    # Count of classified inputs by Classificators graph
    st.markdown("### Number of classified inputs by Classificators")

    input_counts = (
        input_df.groupby(["classificator", "classification"])
        .size()
        .reset_index(name="count")
    )

    chart1 = (
        alt.Chart(input_counts)
        .mark_bar()
        .encode(
            x=alt.X("classificator:N", title="Classificator",
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("classification:N", title="Input Type"),
            y=alt.Y("count:Q", title="Quantity"),
            color=alt.Color(
                "classification:N",
                scale=alt.Scale(domain=["JAILBREAK", "BENIGN"],
                                range=["red", "green"]),
                legend=alt.Legend(title="Input Type")
            ),
            tooltip=[
                alt.Tooltip("classification:N", title="Input Type"),
                alt.Tooltip("count:Q", title="Quantity"),
                alt.Tooltip("classificator:N", title="Classificator")
        ]
        )
    )

    st.altair_chart(chart1, use_container_width=True)

    # Count of classified inputs by Classificators graph

    st.markdown("### Number of classified inputs by SLMRAG")

    slmrag_output = output_df[output_df["classificator"] == "SLMRAG"]
    slmrag_counts = slmrag_output["classification"].value_counts().reset_index()
    slmrag_counts.columns = ["classification", "count"]

    chart2 = (
        alt.Chart(slmrag_counts)
        .mark_bar()
        .encode(
            x=alt.X("classification:N", title="Classification",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Quantity"),
            color=alt.Color("classification:N",
                            scale=alt.Scale(domain=["TOXIC", "SAFE"],
                                            range=["red", "green"]),
                                            legend=alt.Legend(title="Output Type")
                            ),
            tooltip=[
                alt.Tooltip("classification:N", title="Output Type"),
                alt.Tooltip("count:Q", title="Quantity"),
                alt.Tooltip("classificator:N", title="Classificator")
        ]
        )
    )

    st.altair_chart(chart2, use_container_width=True)

    # Input Events over Time

    st.markdown("### Input Events over Time")

    input_df["timestamp"] = pd.to_datetime(input_df["timestamp"])

    option = st.selectbox(
        "Time Granularity",
        ["Day", "Month", "Year"]
    )

    if option == "Day":
        available_days = input_df["timestamp"].dt.date.unique()
        selected_day = st.selectbox("Select Day", sorted(available_days))
        
        day_df = input_df[input_df["timestamp"].dt.date == selected_day].copy()
        day_df["period"] = day_df["timestamp"].dt.hour  # 0–23 horas
        
        counts = (
            day_df.groupby(["period", "classification"])
            .size()
            .reset_index(name="count")
            .sort_values("period")
        )
        
        x_enc = alt.X("period:O", title="Hour of Day", sort=list(range(24)))

    elif option == "Month":
        available_months = input_df["timestamp"].dt.to_period("M").unique()
        selected_month = st.selectbox("Select Month", sorted(available_months.astype(str)))
        
        month_df = input_df[input_df["timestamp"].dt.to_period("M").astype(str) == selected_month].copy()
        
        month_df["bin_id"] = ((month_df["timestamp"].dt.day - 1) // 5) + 1  # 1,2,3,4,...
        month_df["period"] = ((month_df["timestamp"].dt.day - 1) // 5) * 5 + 1
        month_df["period"] = month_df["period"].astype(str) + "-" + (month_df["period"]+4).astype(str)

        counts = (
            month_df.groupby(["bin_id","period","classification"])
            .size()
            .reset_index(name="count")
            .sort_values("bin_id")
        )

        x_enc = alt.X("period:N", title="Day Groups", sort=counts["bin_id"].unique().tolist())



    elif option == "Year":
        available_years = input_df["timestamp"].dt.year.unique()
        selected_year = st.selectbox("Select Year", sorted(available_years))
        
        year_df = input_df[input_df["timestamp"].dt.year == selected_year].copy()
        year_df["period"] = year_df["timestamp"].dt.month
        
        counts = (
            year_df.groupby(["period", "classification"])
            .size()
            .reset_index(name="count")
            .sort_values("period")
        )
        
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        x_enc = alt.X(
            "period:O",  # categórico ordenado
            title="Month",
            sort=list(range(1,13)),  # orden correcto
            axis=alt.Axis(labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]")
        )


    chart3 = (
        alt.Chart(counts)
        .mark_line(point=True)
        .encode(
            x=x_enc,
            y=alt.Y("count:Q", title="Quantity"),
            color=alt.Color("classification:N",
                            scale=alt.Scale(domain=["JAILBREAK", "BENIGN"],
                                            range=["red", "green"])),
            tooltip=["period:N", "classification:N", "count:Q"]
        )
    )

    st.altair_chart(chart3, use_container_width=True)

