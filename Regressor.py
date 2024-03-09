import csv

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

def write_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')

    with open(file_path, 'w', newline='', encoding='iso-8859-1') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def process_time_series_data(file_2006, file_2011, file_2016, file_2022):
    # Read data from CSV files
    df_2006 = pd.read_csv(file_2006, encoding='iso-8859-1')
    df_2011 = pd.read_csv(file_2011, encoding='iso-8859-1')
    df_2016 = pd.read_csv(file_2016, encoding='iso-8859-1')
    df_2022 = pd.read_csv(file_2022, encoding='iso-8859-1')

    # Process 2006 data
    df_2006['DailySpeakers'] = df_2006['Speaks Irish daily within and also outside education '] + df_2006[
        'Outside education system, daily']
    df_2006.rename(columns={'TotalPopulation': 'Population'}, inplace=True)
    df_2006['ProportionDailySpeakers'] = df_2006['DailySpeakers'] / df_2006['Population']

    # Process 2011 data (add daily within and outside education + daily only outside education
    df_2011['DailySpeakers'] = df_2011['T3_3DT'] + df_2011['T3_2DO']
    df_2011.rename(columns={'T3_1T': 'Population'}, inplace=True)
    df_2011['ProportionDailySpeakers'] = df_2011['DailySpeakers'] / df_2011['Population']

    # Process 2016 data (columns renamed from 2011)
    df_2016['DailySpeakers'] = pd.to_numeric(df_2016['T3_2DIDOT']) + pd.to_numeric(
        df_2016['T3_2DOEST'].str.replace(',', ''), errors='coerce')
    df_2016.rename(columns={'T3_1T': 'Population'}, inplace=True)
    df_2016['ProportionDailySpeakers'] = df_2016['DailySpeakers'] / pd.to_numeric(
        df_2016['Population'].str.replace(',', ''), errors='coerce')

    # Process 2022 data (same column names as 2016)
    df_2022['DailySpeakers'] = pd.to_numeric(df_2022['T3_2DIDOT']) + pd.to_numeric(df_2022['T3_2DOEST'],
                                                                                   errors='coerce')
    df_2022.rename(columns={'T3_1T': 'Population'}, inplace=True)
    df_2022['ProportionDailySpeakers'] = df_2022['DailySpeakers'] / pd.to_numeric(df_2022['Population'],
                                                                                  errors='coerce')
    return df_2006, df_2011, df_2016, df_2022


def regression(dataframe):
    # Extract features (X) and target (y) for training
    features = pd.DataFrame()
    features["year1"] = dataframe["Speakers2006"]
    features["year2"] = dataframe["Speakers2011"]
    features["year3"] = dataframe["Speakers2016"]

    targets = pd.DataFrame()
    targets["year4"] = dataframe["Speakers2022"]

    # Extract features (X) for prediction
    future_features = pd.DataFrame()
    future_features["year1"] = dataframe["Speakers2011"]
    future_features["year2"] = dataframe["Speakers2016"]
    future_features["year3"] = dataframe["Speakers2022"]

    # Initialize MLPRegressor
    regressor = MLPRegressor(random_state=42)
    regressor.fit(features, targets)
    # Predict outputs using the provided features
    dataframe['Speakers2027'] = regressor.predict(future_features)
    #print(dataframe)
    write_csv(dataframe, "Regressor_Predict.csv")

def combineDataFrames(df2006, df2011, df2016, df2022):
    # Create a new DataFrame with GEOGDESC column from df2022
    combinedDF = pd.DataFrame(df2022["GEOGDESC"])
    combinedDF.rename(columns={"GEOGDESC": "Electoral District"}, inplace=True)
    # Add GUID column from df2022
    combinedDF["GUID"] = df2022["GUID"]

    # Add ProportionDailySpeakers columns from each DataFrame
    combinedDF["Speakers2006"] = df2006["ProportionDailySpeakers"]
    combinedDF["Speakers2011"] = df2011["ProportionDailySpeakers"]
    combinedDF["Speakers2016"] = df2016["ProportionDailySpeakers"]
    combinedDF["Speakers2022"] = df2022["ProportionDailySpeakers"]

    return combinedDF

# Usage
DF2006, DF2011, DF2016, DF2022 = process_time_series_data("SAPS_2006.csv", "SAPS_2011.csv", "SAPS_2016.csv",
                                                          "SAPS_2022.csv")
# combine data sets into a single dataframe
combined_df = combineDataFrames(DF2006, DF2011, DF2016, DF2022)

regression(combined_df)

