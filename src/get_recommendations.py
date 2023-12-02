import subprocess
import argparse
from keras.models import load_model
import pandas as pd
import json
import numpy as np

def run_similarity_script(user_id, num_similar_users):
    """
       Run the similarity script to find similar users based on the given user ID.

       Args:
           user_id (int): The user ID for whom to find similar users.
           num_similar_users (int): The number of similar users to find.
       """
    script_path = "src/sim_users.py"
    subprocess.run(["python", script_path, str(user_id), str(num_similar_users)])
    print(f"Similarity script executed for user {user_id}")

def load_recommendation_model():
    """
    Load the recommendation model from the specified file.

    Returns:
        The loaded recommendation model.
    """
    rec_model = load_model('models/rec_films.h5')
    print("Recommendation model loaded")
    return rec_model

def load_and_process_data():
    """
    Load and preprocess the rating and movie data.

    Returns:
        ratings_df (pd.DataFrame): The preprocessed DataFrame containing rating data.
        movie_df (pd.DataFrame): The DataFrame containing movie data.
    """
    ratings_df = pd.read_csv("data/raw/rating.csv")
    movie_df = pd.read_csv("data/raw/movies.csv")
    print("Data loaded and processed")
    return ratings_df, movie_df

def generate_recommendations(rec_model, client_id, num_films, ratings_df, movie_df):
    """
    Generate movie recommendations for a given user.

    Args:
        rec_model: The recommendation model.
        client_id (int): The user ID for whom to generate recommendations.
        num_films (int): The number of films to recommend.
        ratings_df (pd.DataFrame): The DataFrame containing rating data.
        movie_df (pd.DataFrame): The DataFrame containing movie data.

    Returns:
        top_movie_rec (pd.DataFrame): The DataFrame with generated movie recommendations.
    """
    movie_watched = ratings_df[ratings_df['userId'] == client_id]['movieId'].values

    json_file_path = f'data/users/similar_users_to_{client_id}.json'

    with open(json_file_path, 'r') as file:
        similar_users_data = json.load(file)

    # Получите список пользователей из json
    similar_users_list = similar_users_data["similar_users"]
    filtered_ratings_df = ratings_df[(ratings_df['userId'].isin(similar_users_list)) & (ratings_df['rating'] >= 3)]
    result_df = pd.merge(filtered_ratings_df, movie_df, on='movieId', how='inner')
    movies_id = result_df["movieId"].unique()
    movie_poll = []
    for item in movies_id:
        if not np.isin(item, movie_watched):
            movie_poll.append(item)

    d = {'userId': [client_id] * len(movie_poll), 'movieId': movie_poll}
    client_df = pd.DataFrame(d)

    ratings = rec_model.predict([client_df['userId'], client_df['movieId']])

    top_ratings_idx = ratings.flatten().argsort()[-num_films:][::-1]
    top_ratings = ratings[top_ratings_idx].flatten()
    recommend_movieId = [movie_poll[x] for x in top_ratings_idx]

    top_movie_rec = (
        pd.DataFrame({'movieId': recommend_movieId, 'prediction': top_ratings})
        .join(result_df.set_index('movieId'), on='movieId')
        .drop_duplicates(subset='movieId', keep='first')
    )
    columns_to_keep = ['title', 'prediction', 'movieId', 'genres', 'userId', 'rating']
    top_movie_rec = top_movie_rec.loc[:, columns_to_keep]
    return top_movie_rec

def save_recommendations(recommendations, user_id):
    """
    Save the generated recommendations to a CSV file.

    Args:
        recommendations (pd.DataFrame): The DataFrame containing recommendations.
        user_id (int): The user ID for whom recommendations were generated.
    """
    output_csv_path = f"data/recommendations/user_{user_id}_rec.csv"
    recommendations.to_csv(output_csv_path, index=False)
    print(f"Recommendations for user {user_id} saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate movie recommendations for a user.')
    parser.add_argument('user_id', type=int, help='User ID for whom to generate recommendations')
    parser.add_argument('num_similar_users', type=int, help='Number of similar users to consider')
    parser.add_argument('num_films', type=int, help='Number of films to recommend')

    args = parser.parse_args()

    user_id = args.user_id
    num_similar_users = args.num_similar_users
    num_films = args.num_films

    run_similarity_script(user_id, num_similar_users)
    rec_model = load_recommendation_model()
    ratings_df, movie_df = load_and_process_data()
    recommendations = generate_recommendations(rec_model, user_id, num_films, ratings_df, movie_df)
    save_recommendations(recommendations, user_id)
