import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set theme
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

st.set_page_config(page_title="India Socioeconomic Dashboard", layout="wide")

st.title("📊 India Socioeconomic Dashboard (Literacy, GDP, Fertility, Population)")

@st.cache_data
def load_data():
    literacy_df = pd.read_excel("litreacy.XLSX", engine="openpyxl", skiprows=3)
    literacy_df = literacy_df.drop(columns=literacy_df.columns[0])
    literacy_df.columns = ['State/UT', '1951', '1961', '1971', '1981', '1991', '2001', '2011']
    literacy_df.dropna(subset=['State/UT'], inplace=True)

    gdp_df = pd.read_excel("GDP.XLSX", engine="openpyxl", skiprows=4)
    if 'Unnamed: 0' in gdp_df.columns:
        gdp_df = gdp_df.drop('Unnamed: 0', axis=1)
    gdp_df.columns = ['State/UT', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09']
    gdp_df.dropna(subset=['State/UT'], inplace=True)

    fertility_df = pd.read_excel("fertility.XLSX", engine="openpyxl", skiprows=2)
    if 'Unnamed: 0' in fertility_df.columns:
        fertility_df = fertility_df.drop('Unnamed: 0', axis=1)
    fertility_df.columns = ['State/UT', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
    fertility_df.dropna(subset=['State/UT'], inplace=True)

    population_df = pd.read_csv("state wise pop.csv")
    population_df.columns = population_df.columns.str.strip()
    population_df = population_df.rename(columns={'population(2011)': 'Population'})

    return literacy_df.reset_index(drop=True), gdp_df.reset_index(drop=True), fertility_df.reset_index(drop=True), population_df

literacy_df, gdp_df, fertility_df, population_df = load_data()

# Sidebar
st.sidebar.title("📁 Data Overview")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Literacy Data")
    st.dataframe(literacy_df)
    st.subheader("GDP Data")
    st.dataframe(gdp_df)
    st.subheader("Fertility Data")
    st.dataframe(fertility_df)
    st.subheader("Population Data")
    st.dataframe(population_df)

# Convert necessary columns to numeric
for col in ['1951', '1961', '1971', '1981', '1991', '2001', '2011']:
    literacy_df[col] = pd.to_numeric(literacy_df[col], errors='coerce')

for col in ['2004-05', '2005-06', '2006-07', '2007-08', '2008-09']:
    gdp_df[col] = pd.to_numeric(gdp_df[col], errors='coerce')

for col in ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']:
    fertility_df[col] = pd.to_numeric(fertility_df[col], errors='coerce')

# Literacy Trend Plot
st.subheader("📚 Literacy Trends Across Years")
fig_lit, ax_lit = plt.subplots()
for _, row in literacy_df.iterrows():
    ax_lit.plot(['1951', '1961', '1971', '1981', '1991', '2001', '2011'], row[1:], label=row['State/UT'])
ax_lit.set_title("Literacy Rate Trends")
ax_lit.set_ylabel("Literacy Rate (%)")
ax_lit.set_xlabel("Year")
ax_lit.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1, 1))
st.pyplot(fig_lit)

# GDP Trend Plot
st.subheader("💰 GDP Trends Across States")
fig_gdp, ax_gdp = plt.subplots()
for _, row in gdp_df.iterrows():
    ax_gdp.plot(['2004-05', '2005-06', '2006-07', '2007-08', '2008-09'], row[1:], label=row['State/UT'])
ax_gdp.set_title("GDP Trends")
ax_gdp.set_ylabel("GDP (Rs. Crore)")
ax_gdp.set_xlabel("Year")
ax_gdp.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1, 1))
st.pyplot(fig_gdp)

# Fertility Trend Plot
st.subheader("🍼 Fertility Rate Trends Across States")
fig_fert, ax_fert = plt.subplots()
for _, row in fertility_df.iterrows():
    ax_fert.plot(row.index[1:], row[1:], label=row['State/UT'])
ax_fert.set_title("Fertility Rate Trends")
ax_fert.set_ylabel("Children per Woman")
ax_fert.set_xlabel("Year")
ax_fert.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1, 1))
st.pyplot(fig_fert)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap (Literacy, GDP, Fertility)")
combined_df = pd.DataFrame()
combined_df['State/UT'] = literacy_df['State/UT']
combined_df['Literacy_2011'] = literacy_df['2011']
combined_df['GDP_2008-09'] = gdp_df['2008-09']
combined_df['Fertility_2011'] = fertility_df['2011']
combined_df.dropna(inplace=True)

corr = combined_df[['Literacy_2011', 'GDP_2008-09', 'Fertility_2011']].corr()
fig_corr, ax_corr = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

# Interactive Plot
st.subheader("💬 Interactive GDP vs Literacy vs Fertility")
fig_plotly = px.scatter(combined_df, x="Literacy_2011", y="GDP_2008-09", color="Fertility_2011",
                        size="GDP_2008-09", hover_name="State/UT",
                        title="GDP vs Literacy vs Fertility (Interactive)")
st.plotly_chart(fig_plotly)

# 3D Scatter Plot
st.subheader("📚💰🧑‍🤝‍🧑 3D View: Literacy vs GDP vs Population")
combined_df = pd.merge(combined_df, population_df[['State/UT', 'Population']], on='State/UT')
fig_3d = px.scatter_3d(combined_df, x="Literacy_2011", y="GDP_2008-09", z="Population",
                       text="State/UT", color="Literacy_2011", size="GDP_2008-09",
                       title="3D View: Literacy, GDP, Population", template='plotly_dark')
st.plotly_chart(fig_3d)
