import pandas as pd

def abilityToSpeakIrish():
    # Read in all 4 datasets
    saps_2006 = pd.read_csv("Output CSVs/SAPS_2006.csv")
    saps_2011 = pd.read_csv("Output CSVs/SAPS_2011.csv")
    saps_2016 = pd.read_csv("Output CSVs/SAPS_2016.csv")
    saps_2022 = pd.read_csv("Output CSVs/SAPS_2022.csv")

    # Create a new dataframe combinedDF
    combinedDF = pd.DataFrame()

    # Remove commas from all columns in all datasets
    saps_2006.replace(',', '', regex=True, inplace=True)
    saps_2011.replace(',', '', regex=True, inplace=True)
    saps_2016.replace(',', '', regex=True, inplace=True)
    saps_2022.replace(',', '', regex=True, inplace=True)

    # Add GUID and GEOGDESC columns from SAPS_2022 to combinedDF
    combinedDF['GUID'] = saps_2022['GUID']
    combinedDF['GEOGDESC'] = saps_2022['GEOGDESC']

    # Process SAPS_2006
    saps_2006['Total'] = pd.to_numeric(saps_2006['Total'])
    saps_2006['TotalPopulation'] = pd.to_numeric(saps_2006['TotalPopulation'])
    combinedDF['Speakers2006'] = saps_2006['Total'] / saps_2006['TotalPopulation']

    # Process SAPS_2011
    saps_2011['T3_1YES'] = pd.to_numeric(saps_2011['T3_1YES'])
    saps_2011['T3_1NO'] = pd.to_numeric(saps_2011['T3_1NO'])
    combinedDF['Speakers2011'] = saps_2011['T3_1YES'] / (saps_2011['T3_1YES'] + saps_2011['T3_1NO'])

    # Process SAPS_2016
    saps_2016['T3_1YES'] = pd.to_numeric(saps_2016['T3_1YES'])
    saps_2016['T3_1NO'] = pd.to_numeric(saps_2016['T3_1NO'])
    combinedDF['Speakers2016'] = saps_2016['T3_1YES'] / (saps_2016['T3_1YES'] + saps_2016['T3_1NO'])

    # Process SAPS_2022
    saps_2022['T3_1YES'] = pd.to_numeric(saps_2022['T3_1YES'])
    saps_2022['T3_1NO'] = pd.to_numeric(saps_2022['T3_1NO'])
    combinedDF['Speakers2022'] = saps_2022['T3_1YES'] / (saps_2022['T3_1YES'] + saps_2022['T3_1NO'])
    # Output combined dataframe to a single file
    combinedDF.to_csv("abilityToSpeakIrish.csv", index=False)
    print(combinedDF.to_string())

def averages():
    regressDF=pd.read_csv("Output CSVs/RegressorPredictionWithError.csv")
    epidemDF = pd.read_csv("Output CSVs/Epidemiology_0101.csv")

    # Calculate the change
    regressDF['change'] = regressDF['Speakers2027'] - regressDF['Speakers2022']
    epidemDF['change'] = epidemDF['Speakers2027'] - epidemDF['Speakers2022']
    # Calculate the mean of the change
    regress_mean_change = regressDF['change'].mean()
    epidem_mean_change = epidemDF['change'].mean()
    # Print the mean of the change
    print("Mean of the Regressor change:", regress_mean_change)
    print("Mean of the Epidemiology model change",epidem_mean_change)

averages()