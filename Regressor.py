import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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


def regression(combined_df):
    # Split features and target variable
    X = combined_df[['CensusNum']].values
    y = combined_df['ProportionDailySpeakers'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize MLPRegressor
    regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

    # Fit the model
    #print(y_train)
    regressor.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = regressor.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Return trained regressor
    return regressor


# Usage
DF2006, DF2011, DF2016, DF2022 = process_time_series_data("SAPS_2006.csv", "SAPS_2011.csv", "SAPS_2016.csv",
                                                          "SAPS_2022.csv")
# combine data sets into a single dataframe
combined_df = pd.concat([DF2006, DF2011, DF2016, DF2022], ignore_index=True)
regressor = regression(combined_df)

# Create a DataFrame with CensusNum = 5 for each row (corresponds to 2027 census)
future_df = pd.DataFrame({'CensusNum': [5] * len(combined_df)})
# Predict the future values of ProportionDailySpeakers
future_prediction = regressor.predict(future_df[['CensusNum']])
# Add the predicted values to the DataFrame
future_df['ProportionDailySpeakers'] = future_prediction
# Print the DataFrame with predicted values
print("Predicted values of ProportionDailySpeakers in 2027:")
print(future_df)

# print("2006")
# print(DF2006)
# print("2011")
# print(DF2011)
# print("2016")
# print(DF2016)
# print("2022")
# print(DF2022['ProportionDailySpeakers'])

