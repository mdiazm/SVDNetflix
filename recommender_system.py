"""
Script to hold code to construct the content-based recommender system.
"""

from data import User
from data import Movie
from data import Reader
import math
import numpy as np
from chrono import Timer
from concurrent.futures import ThreadPoolExecutor

class CollaborativeFilteringRS:


    """
    Class to calculate TF-IDF of the collection.
    """
    def __init__(self):
        self.tag_movie = {}
        self.initialized = False

        self.movie_similarities = {}

    def initialize_system(self):
        if not self.initialized:
            # Read data directly from this class
            reader = Reader()
            reader.write_serialized()
            users, movies = reader.load_serialized()

            # Store data in class attributes
            self.users = users
            self.movies = movies

            # Get ids of the movies in order to calculate similitude in next stages
            self.list_movies_id = list(self.movies)

            # Calculate similar movies
            with Timer() as time:
                self.calculate_similarities()

            print("Ellapsed time calculating similarities: {} seconds".format(time.elapsed))

            # Control initialization of the system
            self.initialized = True

    def pearson(self, item_a, item_b):
        """
        Function to calculate Pearson Coefficient of Correlation between 2 items (movies)
        :param item_a:
        :param item_b:
        :return: Pearson Coefficient of Correlation for items a and b.
        """
        # Get ratings for each item
        item_a_ratings = self.movies[item_a].get_ratings()
        item_b_ratings = self.movies[item_b].get_ratings()

        # Get users who rated each item
        users_a = [int(item[0]) for item in item_a_ratings]
        users_b = [int(item[0]) for item in item_b_ratings]

        # Only common users
        common_users = set(set(users_a) & set(users_b))

        # Get list of evaluations for each item (ONLY TAKING INTO ACCOUNT COMMON USERS)
        ratings_a = {int(item[0]): item[1] for item in item_a_ratings if item[0] in common_users}
        ratings_b = {int(item[0]): item[1] for item in item_b_ratings if item[0] in common_users}

        ratings_a = {int(item[0]): float(item[1]) for item in item_a_ratings}
        ratings_b = {int(item[0]): float(item[1]) for item in item_b_ratings}

        # Get average rating for each item
        avg_a = np.mean([ratings_a[item] for item in ratings_a])
        avg_b = np.mean([ratings_b[item] for item in ratings_b])

        # Calculate numerator of the Pearson Coefficient of Correlation
        numerator = np.sum([(ratings_a[user] - avg_a) * (ratings_b[user] - avg_b) for user in common_users])

        # Calculate denominator
        denominator = np.sqrt(np.sum([(ratings_a[user] - avg_a) ** 2 for user in common_users])) * np.sqrt(np.sum([(ratings_b[user] - avg_b) ** 2 for user in common_users]))

        # Calculate final similitude
        similitude = numerator / denominator

        return similitude

    def calculate_similarities(self):
        """
        Calculate similitude between each pair of movies in the database
        """

        # Get data from pickle file
        data = Reader.load_similarities(None)
        if data is not None:
            self.movie_similarities = data
            return None

        # Execute this portion of code independently
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Select one movie
            for i, movie in enumerate(self.list_movies_id):
                # Evaluate the left ones
                # One thread per movie
                executor.submit(self.similarities_movie, i, movie)

        # Write data to pickle file
        Reader.write_similarities(None, self.movie_similarities)

    def similarities_movie(self, i, movie):
        """
        Calculate similitude between movie with id movie and each other movie in the database.

        :param i: index of the movie in the array self.list_movies_id
        :param movie: id of the movie
        """
        other_movies = self.list_movies_id[:i] + self.list_movies_id[i+1:]  # This gives every movie on list except the selected one

        # Initialize list
        self.movie_similarities[movie] = {}

        # Neighbors
        neighbors = []

        for i, another_movie in enumerate(other_movies):
            similitude = self.pearson(movie, another_movie)
            # self.movie_similarities[movie][another_movie] = similitude if similitude > 0 else 0
            # Discard non-positive similarities
            if similitude >= 0:
                neighbors.append((another_movie, similitude))

        # Sort list
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # Trunk list according to max neighbors
        if len(neighbors) > 50:
            neighbors = neighbors[:50]

        # Fill self.movie_similarities object
        for (another_movie, similitude) in neighbors:
            self.movie_similarities[movie][another_movie] = similitude

        print("Movie: {} is ok".format(self.movies[movie].get_title()))


    def calculate_user_preferences(self, id_user):
        """
        Calculate scores for non-seen movies

        :param id_user:
        :return:
        """
        # Initialize scores object
        scores = []

        # Get movies seen by the user
        seen = {id_movie[0]: id_movie[1] for id_movie in self.users[id_user].get_ratings()}

        # Get non-seen movies
        not_seen = list(set(self.list_movies_id) - set(seen))

        # Calculate scores for each non-seen movie
        for not_seen_movie in not_seen:

            # Compute the k most similar high-rated movies
            high_rated_movies = [] # This will be the set S
            for movie in self.movie_similarities[not_seen_movie]:
                high_rated_movies.append((movie, self.movie_similarities[not_seen_movie][movie]))

            # Sort vector according to similarity
            high_rated_movies = sorted(high_rated_movies, key=lambda x: x[1], reverse=True)

            # Intersection of seen movies and high rated movies
            high_rated_movies = [item for item in high_rated_movies if item[0] in seen]
            # Trunk this vector
            high_rated_movies = high_rated_movies[:10]  # K = 10

            summa_num = np.sum([seen[movie[0]] * self.movie_similarities[not_seen_movie][movie[0]] for movie in high_rated_movies])  # user_rating * similitude(seen, not seen) | Numerator
            summa_den = np.sum([np.abs(self.movie_similarities[not_seen_movie][movie[0]]) for movie in high_rated_movies])  # Denominator

            score = summa_num / summa_den

            scores.append((not_seen_movie, score))

        return scores


    def make_recommendation(self, user_id, query_limit=10):
        """
        Method to get movies recommendations of the system.

        :param: user_id: user who is receiving recommendations.
        :query_limit: default 10: limit of results of the query.

        return: list of (id_movie, title, score) of size query_limit.
        """

        # Check system was initialized
        assert self.initialized

        # Get user object
        user = self.users[user_id]

        # Get ranking for the whole collection of movies
        collection = self.calculate_user_preferences(user_id)

        # Final Ranking
        ranking = sorted(collection, key=lambda x: x[1], reverse=True)

        # Return score
        if query_limit > len(ranking):
            return ranking
        else:
            return ranking[:query_limit]

    def query(self, user_id, query_limit=10):
        """
        Method to print a list of items in a specified format (Ranking)

        return: string containing ranking given by the system.
        """

        # Check user exists
        if user_id not in self.users:
            return "User with id: {} not exists".format(user_id)

        if query_limit < 1:
            return "Query limit must be a positive number"

        ranking = self.make_recommendation(int(user_id), query_limit)

        if len(ranking) == 0:
            return "No movies to recommend"

        output = ""
        for i, movie in enumerate(ranking):
            output += "{}. {}. {}: {:.3f}\n".format(i+1, movie[0], self.movies[movie[0]].get_title(), movie[1])

        return output