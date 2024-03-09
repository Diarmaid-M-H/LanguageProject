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

df = pd.read_csv("CombinedDailySpeakers.csv")

# Function to optimize parameters for a single row
def optimize_parameters(Speakers_previous, Speakers_target):
    initial_guess = [0.1, 0.5]  # Initial guess for parameters
    result = minimize(objective_function, initial_guess, args=(Speakers_previous, Speakers_target), bounds=[(0, 1), (0, 1)])
    optimized_params = result.x
    predicted = run_language_model(optimized_params[0], optimized_params[1], Speakers_previous, num_iterations=5)
    error = np.sqrt(np.mean((predicted - Speakers_target)**2))  # Calculating RMSE
    return optimized_params, error

# Load the dataset
df = pd.read_csv("CombinedDailySpeakers.csv")

# Initialize arrays to store optimized parameters for each row
optimized_abandonment_fractions = []
optimized_probs_acquisition = []
#error_array = []

startTime = time.time()
# Use multithreading to optimize parameters for each row
# multi threading is thousands of times faster than single threading for this, basically required.
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(optimize_parameters, row["Speakers2016"], row["Speakers2022"]) for _, row in df.iterrows()]
    for future in futures:
        optimized_params, rmse = future.result()
        optimized_abandonment_fractions.append(optimized_params[0])
        optimized_probs_acquisition.append(optimized_params[1])
        #error_array.append(rmse)
        print("Time elapsed: ", time.time() - startTime)
        print("Optimized parameters: ", optimized_params)
        #print("Error", rmse)

# Add optimized parameters to the dataframe
df["OptimizedAbandonmentFraction"] = optimized_abandonment_fractions
df["OptimizedProbAcquisition"] = optimized_probs_acquisition
#df["RMSE"] = error_array

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Call run_language_model with parameters from the current row
    predicted_speakers_2027 = run_language_model(row["OptimizedProbAcquisition"], row["OptimizedAbandonmentFraction"], row["Speakers2022"], num_iterations=5)
    # Add the result to the dataframe
    df.at[index, "Speakers2027"] = predicted_speakers_2027

# Reorder columns
df = df.reindex(columns=['Speakers2006', 'Speakers2011', 'Speakers2016', 'Speakers2022', 'Speakers2027', 'OptimizedAbandonmentFraction', 'OptimizedProbAcquisition'])

# Print updated dataframe
print(df.to_string())

# Write the updated dataframe to a CSV file
df.to_csv("EpidemiologyModelParameters.csv", index=False)


