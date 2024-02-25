import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


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
    # TODO: rename "name" column of each dataset to the same thing
    # Add census number
    df_2006['CensusNum'] = 1
    df_2011['CensusNum'] = 2
    df_2016['CensusNum'] = 3
    df_2022['CensusNum'] = 4
    return df_2006, df_2011, df_2016, df_2022


def regression(dataframe):
    # Extract features (X) and target (y) for training
    X_train = pd.DataFrame()
    X_train["year1"] = dataframe["Speakers2006"]
    X_train["year2"] = dataframe["Speakers2011"]
    X_train["year3"] = dataframe["Speakers2016"]
    #X_train.rename(columns={'Speakers2006': 'year1','Speakers2011': 'year2','Speakers2016': 'year3'})
    print(X_train.to_string())

    y_train = pd.DataFrame()
    y_train["year4"] = dataframe["Speakers2022"]

    # Extract features (X) for prediction
    X_predict = pd.DataFrame()
    X_predict["year1"] = dataframe["Speakers2011"]
    X_predict["year2"] = dataframe["Speakers2016"]
    X_predict["year3"] = dataframe["Speakers2022"]
    # Initialize MLPRegressor
    regressor = MLPRegressor(random_state=42)

    # Train the model
    regressor.fit(X_train, y_train)

    # Predict outputs using the provided features
    dataframe['Speakers2027'] = regressor.predict(X_predict)
    print(dataframe.to_string())

    # Split data into training and testing sets (80% train, 20% test)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2,
                                                                                random_state=42)
    # Test the model using the test set
    y_pred = regressor.predict(X_test_split)

    # Calculate Mean Squared Error on the test set
    mse_test = mean_squared_error(y_test_split, y_pred)

    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

    # Convert scores to positive as cross_val_score returns negative values
    cv_scores = -cv_scores

    # Calculate mean and standard deviation of cross-validation scores
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)

    # Print the results
    print("Mean Squared Error on Test Set:", mse_test)
    print("Mean of Cross-Validation Scores:", mean_cv_score)
    print("Standard Deviation of Cross-Validation Scores:", std_cv_score)

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


    #print(combinedDF.to_string())

    return combinedDF

# Usage
DF2006, DF2011, DF2016, DF2022 = process_time_series_data("SAPS_2006.csv", "SAPS_2011.csv", "SAPS_2016.csv",
                                                          "SAPS_2022.csv")
# combine data sets into a single dataframe
combined_df = combineDataFrames(DF2006, DF2011, DF2016, DF2022)
regression(combined_df)

# print("2006")
# print(DF2006)
# print("2011")
# print(DF2011)
# print("2016")
# print(DF2016)
# print("2022")
# print(DF2022['ProportionDailySpeakers'])

