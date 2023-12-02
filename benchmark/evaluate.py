import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, ndcg_score
from keras.models import load_model

def mean_average_precision(true_ratings, predicted_ratings):
    """
    Calculate Mean Average Precision.

    Args:
        true_ratings (numpy.ndarray): True ratings from the test set.
        predicted_ratings (numpy.ndarray): Predicted ratings from the model.

    Returns:
        float: Mean Average Precision score.
    """
    threshold = 3.5
    true_items = (true_ratings >= threshold).astype(int)
    predicted_items = (predicted_ratings >= threshold).astype(int)

    map_score = average_precision_score(true_items, predicted_items, average='micro')
    return map_score


def normalized_discounted_cumulative_gain(true_ratings, predicted_ratings):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        true_ratings (numpy.ndarray): True ratings from the test set.
        predicted_ratings (numpy.ndarray): Predicted ratings from the model.

    Returns:
        float: NDCG score.
    """
    threshold = 3.5
    true_items = (true_ratings >= threshold).astype(int)
    predicted_items = (predicted_ratings >= threshold).astype(int)

    ndcg_score_value = ndcg_score([true_items], [predicted_items])
    return ndcg_score_value


def load_and_evaluate_model(model_path, test_data):
    """
    Load a pre-trained model, make predictions on test data,
    calculate evaluation metrics, and log the results.

    Args:
        model_path (str): The path to the pre-trained model file.
        test_data (pd.DataFrame): The DataFrame containing test data.

    """
    model = load_model(model_path)
    test_predictions = model.predict([test_data.userId, test_data.movieId])
    mean_ap = mean_average_precision(test_data.rating.values, test_predictions.flatten())
    ndcg = normalized_discounted_cumulative_gain(test_data.rating.values, test_predictions.flatten())
    results_df = pd.DataFrame({
        'Metric': ['MAP', 'NDCG'],
        'Value': [mean_ap, ndcg]
    })

    print(results_df)
    plt.bar(results_df['Metric'], results_df['Value'], color=['blue', 'green'])
    plt.ylabel('Metric Value')
    plt.title('Model Evaluation Metrics')
    plt.savefig('evaluation_results.png')
    results_df.to_csv('evaluation_results.csv', index=False)


if __name__ == "__main__":
    model_path = '../models/rec_films.h5'
    test_data = pd.read_csv("../benchmark/data/evaluate.csv")
    load_and_evaluate_model(model_path, test_data)
