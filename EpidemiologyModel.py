import csv
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import numpy as np


class LanguageModel:
    def __init__(self, initial_irish_speaking, initial_english_speaking, prob_acquisition, abandonment_fraction):
        self.irish_speaking = [initial_irish_speaking]
        self.english_speaking = [initial_english_speaking]
        self.prob_acquisition = prob_acquisition
        self.abandonment_fraction = abandonment_fraction

    def update(self):
        prob_meeting = self.irish_speaking[-1]  # Prob meeting is based on current proportion of Irish speakers

        learning_rate = self.prob_acquisition * prob_meeting * self.english_speaking[-1]
        abandonment_rate = self.abandonment_fraction * self.irish_speaking[-1]

        # Update stocks
        new_irish = self.irish_speaking[-1] + learning_rate - abandonment_rate
        new_english = 1 - new_irish

        # Ensure proportions sum to 1
        self.irish_speaking.append(new_irish)
        self.english_speaking.append(new_english)
def write_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')

    with open(file_path, 'w', newline='', encoding='iso-8859-1') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def run_language_model(prob_acquisition, abandonment_fraction, initial_irish_speaking, num_iterations):
    initial_english_speaking = 1 - initial_irish_speaking

    # Create the model
    model = LanguageModel(initial_irish_speaking, initial_english_speaking, prob_acquisition, abandonment_fraction)

    # Update the model for a certain number of iterations
    for _ in range(num_iterations):
        model.update()

    # # Plot the results
    # plt.plot(model.irish_speaking,
    #          label=f'Prob Acquisition={prob_acquisition}, Abandonment Fraction={abandonment_fraction}')
    # # print("Final Proportion of Irish Speaking Population:", model.irish_speaking[-1])
    # # Plot formatting
    # plt.xlabel('Iterations')
    # plt.ylabel('Proportion Irish Speakers')
    # plt.title('Language Dynamics')
    # plt.legend()
    # #plt.show()

    return model.irish_speaking[-1]

# use final_irish_speaking[-1] to get last entry
# Function to calculate the difference between predicted and actual values
def objective_function(params, initial_irish_speaking, target_irish_speaking):
    prob_acquisition, abandonment_fraction = params
    predicted = run_language_model(prob_acquisition, abandonment_fraction, initial_irish_speaking, num_iterations=5)
    error = np.sqrt(np.mean((predicted - target_irish_speaking)**2))  # Calculating Root Mean Squared Error
    return error


# Function to optimize parameters for a single row
def optimize_parameters(Speakers_previous, Speakers_target,initial_guess):
    # initial_guess = [0.2, 0.1]  # Initial guess for parameters
    result = minimize(objective_function, initial_guess, args=(Speakers_previous, Speakers_target), bounds=[(0, 1), (0, 1)])
    optimized_params = result.x
    predicted = run_language_model(optimized_params[0], optimized_params[1], Speakers_previous, num_iterations=5)
    return optimized_params


def parameter_Optimization(df, paramBefore, paramAfter,initial_guess):
    # Initialize arrays to store optimized parameters for each row
    optimized_abandonment_fractions = []
    optimized_probs_acquisition = []

    startTime = time.time()

    # Use multithreading to optimize parameters for each row
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(optimize_parameters, row[paramBefore], row[paramAfter], initial_guess) for _, row in df.iterrows()]
        for future in futures:
            optimized_params = future.result()
            optimized_abandonment_fractions.append(optimized_params[0])
            optimized_probs_acquisition.append(optimized_params[1])
            # print("Time elapsed: ", time.time() - startTime)
            # print("Optimized parameters: ", optimized_params)

    return optimized_abandonment_fractions, optimized_probs_acquisition

def parameterGridSearch(df):
    # Define initial guesses for parameters
    prob_acquisition_values = [0.01,0.015,0.02,0.025,0.03]
    abandonment_fraction_values = [0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005]

    # Initialize dictionaries to store mean absolute errors and standard deviations
    mean_absolute_errors = {}
    std_dev_errors = {}

    # Loop through the initial guesses
    for prob_acquisition in prob_acquisition_values:
        for abandonment_fraction in abandonment_fraction_values:
            # Set initial guess
            initial_guess = [prob_acquisition, abandonment_fraction]

            # Perform optimization
            optimized_abandonment_fractions, optimized_probs_acquisition = parameter_Optimization(df, "Speakers2011",
                                                                                                  "Speakers2016",
                                                                                                  initial_guess)

            # Make prediction for 2022 using optimized parameters
            prediction2022 = makePrediction(df, "Speakers2016", optimized_abandonment_fractions,
                                            optimized_probs_acquisition)

            # Calculate absolute differences
            absolute_differences = [abs(actual - predicted) for actual, predicted in
                                    zip(df['Speakers2022'].tolist(), prediction2022)]

            # Calculate mean error and standard deviation of error
            mean_error = np.mean(absolute_differences)
            std_dev_error = np.std(absolute_differences)

            # Store mean error and standard deviation in dictionaries
            if prob_acquisition not in mean_absolute_errors:
                mean_absolute_errors[prob_acquisition] = {}
                std_dev_errors[prob_acquisition] = {}
            mean_absolute_errors[prob_acquisition][abandonment_fraction] = mean_error
            std_dev_errors[prob_acquisition][abandonment_fraction] = std_dev_error

            print(initial_guess,"calculation complete")

    # Write mean absolute errors to CSV file
    with open('mean_absolute_errors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #abandonment fraction along the top.
        writer.writerow(["Prob_Acquisition"] + abandonment_fraction_values)
        #prob acquisition along the side
        for prob_acquisition in prob_acquisition_values:
            writer.writerow([prob_acquisition] + [mean_absolute_errors[prob_acquisition][af] for af in abandonment_fraction_values])

    # Write standard deviations to CSV file
    with open('std_dev_errors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prob_Acquisition"] + abandonment_fraction_values)
        for prob_acquisition in prob_acquisition_values:
            writer.writerow([prob_acquisition] + [std_dev_errors[prob_acquisition][af] for af in abandonment_fraction_values])

def makePrediction(df,startYear,prob_abandonment,prob_acquisition):
    predictionArray = []
    # Iterate over each row in the dataframe to make prediction
    # predicts 5 years ahead.
    for index, row in df.iterrows():
        # Call run_language_model with parameters from the current row
        prediction = run_language_model(prob_acquisition[index],
                                                     prob_abandonment[index], row[startYear],
                                                     num_iterations=5)
        # Add the result to the dataframe
        predictionArray.append(prediction)
    return predictionArray

def langModel(df):

    # firstly, optimize parameters with 2011 as the start time and 2016 as end time
    initial_guess = [0.02, 0.0000005]
    optimized_abandonment_fractions,optimized_probs_acquisition = parameter_Optimization(df,"Speakers2011","Speakers2016",initial_guess)
    # using the optimized parameters, make a prediction of 2022 values with 2016 as the start time
    prediction2022 = makePrediction(df,"Speakers2016",optimized_abandonment_fractions,optimized_probs_acquisition)
    #print(prediction2022)
    actual_speakers = df['Speakers2022'].tolist()

    # compare prediction of 2022 with actual 2022 data.
    # Initialize an empty list to store the absolute differences
    absolute_differences = []

    # Iterate through the predictions and actual values simultaneously
    for actual, predicted in zip(actual_speakers, prediction2022):
        # Calculate the absolute difference and append it to the list
        absolute_difference = abs(actual - predicted)
        absolute_differences.append(absolute_difference)

    # Add absolute_differences as a new column to the DataFrame df
    df['Error'] = absolute_differences

    mean_error = df['Error'].mean()
    std_dev_error = df['Error'].std()

    # Find index of row with lowest error
    lowest_error_index = df['Error'].idxmin()
    # Find index of row with highest error
    highest_error_index = df['Error'].idxmax()

    print()
    # Print entire row with lowest error
    print("Row with Lowest Error:")
    print(df.loc[lowest_error_index])

    # Print entire row with highest error
    print("\nRow with Highest Error:")
    print(df.loc[highest_error_index])
    print("Mean Absolute Error:", mean_error)
    print("Standard Deviation of Error:", std_dev_error)

    # optimize parameters for 2027 prediction
    optimized_abandonment_fractions_prediction, optimized_probs_acquisition_prediction = parameter_Optimization(df, "Speakers2016", "Speakers2022",initial_guess)
    # Add optimized parameters to the dataframe
    predictionArray = makePrediction(df, "Speakers2022",optimized_abandonment_fractions_prediction,optimized_probs_acquisition_prediction)
    df["Speakers2027"] = predictionArray
    df["OptimizedAbandonmentFraction"] = optimized_abandonment_fractions_prediction
    df["OptimizedProbAcquisition"] = optimized_probs_acquisition_prediction
    # Reorder columns
    df = df.reindex(
        columns=['Electoral District', 'GUID', 'Speakers2006', 'Speakers2011', 'Speakers2016', 'Speakers2022',
                 'Speakers2027', 'OptimizedAbandonmentFraction', 'OptimizedProbAcquisition','Error'])

    # Print updated dataframe
    print(df.to_string())

    # Write the updated dataframe to a CSV file
    df.to_csv("DemoEpidemiology.csv", index=False)


dataframe = pd.read_csv("CombinedDailySpeakers.csv")
#parameterGridSearch(dataframe)
langModel(dataframe)



