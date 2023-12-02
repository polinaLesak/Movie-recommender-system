# Movie Recommender System

## Personal info
Polina Lesak, p.lesak@innopolis.university, BS21-DS-01(exchange student)

## Solution

The repository contains code for movie recommender system. A detailed report on the algorithms can be found in the reports/Final_report.pdf

The idea:
1. Using cousin similarity find "similar" users (notebooks/2-knn-user-similarity)
2. Predict recommendations to the user using the NCF model based on movies that were watched by "similar" users(notebooks/3-nfc-recommender)

## Main processes

### Installation

First clone repository:

`git clone https://github.com/polinaLesak/Movie-recommender-system.git`

Install requirements:

`pip install -r requirements.txt`

### Find "similar" users

You can simply find "similar" users using this script:

`python src/sim_users.py <USER_ID> <NUM_OF_USERS_TO_FIND>`

Similar users saved to data/users/similar_users_to_{USER_ID}.json

Example"

`python src/sim_users.py 778 10`

Similar users saved to data/users/similar_users_to_778.json

### Get top N recommendations

To get movie recommendations run this script. 

You ***don't need*** to run sim_users.py before that.

`python src/get_recommendations.py <USER_ID> <NUM_OF_USERS_TO_FIND> <NUM_FILMS_TO_PREDICT>`

Example:

`python src/get_recommendations.py 775 10 5`

Saving similar users to data/users/similar_users_to_775.json

{"similar_users": [784, 788, 745, 772, 769, 855, 808, 829, 852, 766]}

Recommendations for user 775 saved to data/recommendations/user_775_rec.csv

| Title                                  | Prediction | MovieId | Genres                                      | UserId | Rating |
|----------------------------------------|------------|---------|---------------------------------------------|--------|--------|
| Star Wars (1977)                       | 4.334508   | 50      | Action, Adventure, Romance, Sci-Fi, War      | 852    | 5      |
| Empire Strikes Back, The (1980)        | 4.328842   | 172     | Action, Adventure, Drama, Romance, Sci-Fi, War | 766    | 3      |
| Princess Bride, The (1987)             | 4.2975836  | 173     | Action, Adventure, Comedy, Romance            | 766    | 4      |
| Schindler's List (1993)                | 4.272797   | 318     | Drama, War                                  | 766    | 5      |
| Some Folks Call It a Sling Blade (1993)| 4.223508   | 963     | Drama, Thriller                             | 788    | 4      |

## Evaluating

MAP and NDCG are commonly used in recommender systems because they consider the ranking aspect and the relevance of items:

| Metric | Value    |
|--------|----------|
| MAP    | 0.680606 |
| NDCG   | 0.957994 |


![Evaluating_results](https://github.com/polinaLesak/Movie-recommender-system/blob/master/benchmark/evaluation_results.png)

MAP = 0.68 is a relatively good score, suggesting that the model is effective in ranking relevant items higher. 

NDCG = 0.957 is quite high, indicating that the model is providing good rankings for relevant items.

To evaluate model run:

`python benchmark/evaluate.py`

Evaluating results saved to benchmark/evaluation_results.csv and benchmark/evaluation_results.png