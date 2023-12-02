# evaluate_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model

def load_and_evaluate_model(model_path, test_data):
    """
    Load a pre-trained model, make predictions on test data,
    calculate evaluation metrics, and log the results.

    Args:
        model_path (str): The path to the pre-trained model file.
        test_data (pd.DataFrame): The DataFrame containing test data.

    Returns:
        None
    """

    # Load the pre-trained model
    model = load_model(model_path)

    # Predictions on the test set
    test_predictions = model.predict([test_data.userId, test_data.movieId])

    # Calculate evaluation metrics
    test_rmse = np.sqrt(mean_squared_error(test_data.rating, test_predictions))
    test_mae = mean_absolute_error(test_data.rating, test_predictions)
    test_r2 = r2_score(test_data.rating, test_predictions)

    # Create a DataFrame for logging results
    results_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R^2 Score'],
        'Value': [test_rmse, test_mae, test_r2]
    })

    # Print results to console
    print(results_df)

    # Plot the results
    plt.bar(results_df['Metric'], results_df['Value'], color=['blue', 'green', 'orange'])
    plt.ylabel('Metric Value')
    plt.title('Model Evaluation Metrics')

    # Save the plot as a PNG file
    plt.savefig('evaluation_results.png')

    # Save the results DataFrame to a CSV file
    results_df.to_csv('evaluation_results.csv', index=False)

if __name__ == "__main__":
    # Specify the path to the pre-trained model and test data
    model_path = '../models/rec_films.h5'

    # Load the test data
    test_data = pd.read_csv("../benchmark/data/evaluate.csv")

    # Evaluate the model and log the results
    load_and_evaluate_model(model_path, test_data)
