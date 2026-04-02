import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Економічний дашборд України", layout="wide")

st.title("📊 Економічний дашборд України")

# ---------------------------
# Функція отримання даних (World Bank API)
# ---------------------------
def load_data():
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "GDP_growth",
        "FP.CPI.TOTL.ZG": "Inflation",
        "SL.UEM.TOTL.ZS": "Unemployment"
    }

    data = {}

    for code, name in indicators.items():
        url = f"http://api.worldbank.org/v2/country/UA/indicator/{code}?format=json"
        response = requests.get(url)
        json_data = response.json()

        values = []
        for item in json_data[1]:
            if item["value"] is not None:
                values.append({
                    "Year": int(item["date"]),
                    name: item["value"]
                })

        df = pd.DataFrame(values)
        df = df.sort_values("Year")
        data[name] = df

    # Об'єднуємо всі показники
    df_final = data["GDP_growth"]
    df_final = df_final.merge(data["Inflation"], on="Year", how="outer")
    df_final = df_final.merge(data["Unemployment"], on="Year", how="outer")

    df_final = df_final.sort_values("Year")
    df_final = df_final.dropna()

    return df_final

df = load_data()

# ---------------------------
# Показ даних
# ---------------------------
st.subheader("📋 Дані")
st.dataframe(df)

# ---------------------------
# Графіки
# ---------------------------
st.subheader("📈 Динаміка показників")

fig = px.line(df, x="Year", y=df.columns[1:], markers=True)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Кореляція
# ---------------------------
st.subheader("🔗 Кореляція показників")

corr = df[["GDP_growth", "Inflation", "Unemployment"]].corr()
st.write(corr)

fig_corr = px.imshow(corr, text_auto=True)
st.plotly_chart(fig_corr)

# ---------------------------
# Прогноз (лінійна регресія)
# ---------------------------
st.subheader("🔮 Прогноз на 6 періодів вперед")

future_years = np.array(range(df["Year"].max() + 1, df["Year"].max() + 7)).reshape(-1, 1)

forecast_data = pd.DataFrame({"Year": future_years.flatten()})

for column in ["GDP_growth", "Inflation", "Unemployment"]:
    model = LinearRegression()

    X = df["Year"].values.reshape(-1, 1)
    y = df[column].values

    model.fit(X, y)
    forecast = model.predict(future_years)

    forecast_data[column] = forecast

# Об'єднання історії + прогнозу
df_all = pd.concat([df, forecast_data], ignore_index=True)

fig_forecast = px.line(df_all, x="Year", y=df_all.columns[1:], markers=True,
                       title="Прогноз економічних показників")

st.plotly_chart(fig_forecast, use_container_width=True)

st.success("✅ Дашборд успішно побудований!")