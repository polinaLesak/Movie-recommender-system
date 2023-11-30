import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_and_close_plot(figure, file_path):
    """
    Save the current figure and close it.

    Args:
        figure: The Matplotlib figure to save.
        file_path (str): The file path to save the figure.
    """
    figure.savefig(file_path)
    plt.close()

def plot_rating_distribution(rating_df, output_folder):
    """
    Plot the distribution of ratings and save the plot.

    Args:
        rating_df (pd.DataFrame): The DataFrame containing rating data.
        output_folder (str): The folder where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=rating_df)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    file_path = os.path.join(output_folder, 'rating_distribution.png')
    save_and_close_plot(plt, file_path)
    print(f"Saved: {file_path}")

def plot_age_distribution(user_df, output_folder):
    """
    Plot the age distribution of users and save the plot.

    Args:
        user_df (pd.DataFrame): The DataFrame containing user data.
        output_folder (str): The folder where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(user_df['age'], bins=30, kde=True)
    plt.title('Age Distribution of Users')
    plt.xlabel('Age')
    plt.ylabel('Count')
    file_path = os.path.join(output_folder, 'age_distribution.png')
    save_and_close_plot(plt, file_path)
    print(f"Saved: {file_path}")

def plot_user_movie_heatmap(rating_df, top_users, output_folder):
    """
    Plot the user-movie ratings heatmap for the top users and save the plot.

    Args:
        rating_df (pd.DataFrame): The DataFrame containing rating data.
        top_users (list): The list of top users.
        output_folder (str): The folder where the plot will be saved.
    """
    top_users = sorted(top_users)[:20]
    filtered_rating_df = rating_df[rating_df['userId'].isin(top_users)]
    plt.figure(figsize=(12, 8))
    heatmap_data = filtered_rating_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    sns.heatmap(heatmap_data, cmap='coolwarm', mask=(heatmap_data == 0))
    plt.title('User-Movie Ratings Heatmap for Top 20 Users')
    file_path = os.path.join(output_folder, 'user_movie_heatmap.png')
    save_and_close_plot(plt, file_path)
    print(f"Saved: {file_path}")

def plot_num_ratings_per_user(rating_df, output_folder):
    """
    Plot the number of ratings per user and save the plot.

    Args:
        rating_df (pd.DataFrame): The DataFrame containing rating data.
        output_folder (str): The folder where the plot will be saved.
    """
    ratings_per_user = rating_df['userId'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.histplot(ratings_per_user, bins=50, kde=False)
    plt.title('Number of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    file_path = os.path.join(output_folder, 'num_ratings_per_user.png')
    save_and_close_plot(plt, file_path)
    print(f"Saved: {file_path}")

def plot_top_movie_genres(movie_df, output_folder):
    """
    Plot the distribution of top movie genres and save the plot.

    Args:
        movie_df (pd.DataFrame): The DataFrame containing movie data.
        output_folder (str): The folder where the plot will be saved.
    """
    genres = movie_df.columns[5:]
    genres_count = movie_df[genres].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=genres_count.values, y=genres_count.index, hue=genres_count.index, palette='viridis', legend=False)
    plt.title('Top Movie Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    file_path = os.path.join(output_folder, 'top_movie_genres.png')
    save_and_close_plot(plt, file_path)
    print(f"Saved: {file_path}")

def main():
    """
    Main function to generate and save various plots based on rating, user, and movie data.
    """
    rating_df = pd.read_csv("../data/raw/rating.csv")
    user_df = pd.read_csv("../data/raw/users.csv")
    movie_cols = ["movieId", "title", "release_date", "video_release_date", "IMDB_URL", "unknown", "Action",
                  "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                  "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movie_df = pd.read_csv("../data/raw/ml-100k/u.item", sep='|', names=movie_cols, encoding='latin1')

    output_folder = "../report/figures"

    plot_rating_distribution(rating_df, output_folder)
    plot_age_distribution(user_df, output_folder)
    plot_user_movie_heatmap(rating_df, rating_df['userId'].unique(), output_folder)
    plot_num_ratings_per_user(rating_df, output_folder)
    plot_top_movie_genres(movie_df, output_folder)

if __name__ == "__main__":
    main()
