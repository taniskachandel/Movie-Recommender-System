# importing libraries
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# importing our dataset
r_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv("ratings.csv", sep='::', names=r_cols)
ratings = ratings.drop(['timestamp'], axis=1)

m_cols = ['movieId', 'title', 'genres']
movies = pd.read_csv("movies.csv", sep='::', names=m_cols, encoding='latin-1')

# making a pivot table to make working on data easier
movie_features = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# since most of the values are 0 we will create a sparse matrix to avoid overflow and wasted memory
sparse_matrix_movie_features = csr_matrix(movie_features.values)

# DATA PREPROCESSING

# look for total number of users and total number of movies
num_users = len(ratings.userId.unique())
num_movies = len(ratings.movieId.unique())


# getting the count of each rating (1-5)
rating_count = pd.DataFrame(ratings.groupby('rating').size(), columns=['count'])

# a lot of the ratings are 0 (movie was not rated by the user)
total_count = num_users * num_movies
zero_ratings = total_count - ratings.shape[0]

total_count_ratings = rating_count.append(pd.DataFrame({'count': zero_ratings}, index=[0.0]),
                                          verify_integrity=True).sort_index()

# number of ratings each movie got
movies_count = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])

# make threshold values
# now we need to take only movies that have been rated at the least 50 times
popularity_threshold = 50
popular_movies = list(set(movies_count.query('count >= @popularity_threshold').index))
unpopular_movies_dropped = ratings[ratings.movieId.isin(popular_movies)]

# get number of ratings given by every user
users_ratings = pd.DataFrame(unpopular_movies_dropped.groupby('userId').size(), columns=['count'])

# only taking users which have rated at least 50 movies into consideration
ratings_threshold = 50
users_considered = list(set(users_ratings.query('count >= @ratings_threshold').index))
users = unpopular_movies_dropped[unpopular_movies_dropped.userId.isin(users_considered)]


# creating a pivot table (movie-user matrix)
movies_users = users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
get_input_movie_index = {movie: i for i, movie in
                         enumerate(list(movies.set_index('movieId').loc[movies_users.index].title))}


# transform matrix to scipy sparse matrix
sparse_matrix_movies_users = csr_matrix(movies_users.values)

# BUILDING THE MODEL

# defining the model
model_knn = NearestNeighbors(metric='cosine', n_neighbors=10)
# fitting the model
model_knn.fit(sparse_matrix_movies_users)


def get_index(index, fav_movie):
    match_tuple = []
    # get match
    for title, movie_index in index.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, movie_index, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[0])
    if not match_tuple:
        print('Oops! No match is found')
        return
    else:
        print('Found possible matches: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


def recommender_function(knn_model, data, index, input_movie, no_recommendations):
    # get input movie index
    print('You have input movie:', input_movie)
    movie_index = get_index(index, input_movie)

    distances, indices = knn_model.kneighbors(data[movie_index], n_neighbors=no_recommendations+1)

    recommendations = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                             key=lambda x: x[1])[:0:-1]
    reversing_keys = {i: m for m, i in index.items()}

    # print recommendations
    print('Recommendations for {}:'.format(input_movie))
    for i, (movie_index, dist) in enumerate(recommendations):
        print('{0}: {1}'.format(i + 1, reversing_keys[movie_index]))


film_name = input("Enter a movie name: ")

recommender_function(knn_model=model_knn, data=sparse_matrix_movies_users, input_movie=film_name,
                     index=get_input_movie_index, no_recommendations=10)
