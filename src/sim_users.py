import sys
import pandas as pd
import numpy as np
import pickle
import json

def load_knn_alg():
    """
    Load the pre-trained KNN model from the specified file path.

    Returns:
        The loaded KNN model.
    """
    file_path = '../models/similar_users_model.pkl'
    print(f"Loading KNN model from {file_path}")
    with open(file_path, 'rb') as model_file:
        knn_alg = pickle.load(model_file)
    print("KNN model loaded successfully")
    return knn_alg

def get_similar_users(knn_alg, df, user_id, n=5):
    """
    Get similar users for a given user using the KNN model.

    Args:
        knn_alg: The pre-trained KNN model.
        df (pd.DataFrame): The DataFrame containing user data.
        user_id (int): The user ID for whom to find similar users.
        n (int): The number of similar users to retrieve.

    Returns:
        similar_users (list): The list of similar user IDs.
    """
    print(f"Getting {n} similar users for user {user_id}")
    knn_input = np.asarray([df.values[user_id - 1]])
    distances, indices = knn_alg.kneighbors(knn_input, n_neighbors=n + 1)
    similar_users = indices.flatten()[1:] + 1
    print(f"Similar users: {similar_users}")
    return similar_users

def save_similar_users(similar_users, file_path):
    """
    Save the list of similar users to a JSON file.

    Args:
        similar_users (list): The list of similar user IDs.
        file_path (str): The path to the file where the similar users will be saved.
    """
    print(f"Saving similar users to {file_path}")
    data = {"similar_users": [int(user) for user in similar_users]}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
    print("Similar users saved successfully")


if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    df = pd.read_csv("../data/raw/preprocessed.csv", index_col='user_id')
    knn_alg = load_knn_alg()
    user_id_to_find = int(sys.argv[1])
    num_similar_users = int(sys.argv[2])
    similar_users = get_similar_users(knn_alg, df, user_id_to_find, num_similar_users)
    file_path = f"../data/users/similar_users_to_{user_id_to_find}.json"
    save_similar_users(similar_users, file_path)

    print(f"Similar users saved to {file_path}")
