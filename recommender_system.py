"""
Script to hold code to construct the content-based recommender system.
"""

from data import Reader
from chrono import Timer
import numpy as np
import pickle
import config as cfg
import os

class SVDNetflix:

    """
    Class to implement algorithm
    """

    # Real data
    ratingsMatrix = None
    userIndexes = None
    moviesIndexes = None
    ratingsTuples = None

    # Singular Value Decomposition parameters
    usersPreferences = None
    moviesPreferences = None
    numLatentFactors = 40
    initializationValue = 0.1
    learningRate = 0.001
    regularizeParameter = 0.02 # As recommended in the article https://sifter.org/~simon/journal/20061211.html
    numEpochs = 120

    def __init__(self):
        self.tag_movie = {}
        self.initialized = False
        self.movie_similarities = {}

        # Cache for avoiding calculating product between feature vectors on each call to predict function.
        self.cache = None

    def initialize_system(self):
        if not self.initialized:
            # Read data directly from this class
            reader = Reader()
            reader.write_serialized()

            users, movies, self.ratingsTuples = reader.load_serialized()
            (self.ratingsMatrix, self.userIndexes, self.moviesIndexes) = reader.create_ratings_matrix(users, movies)

            # Store data in class attributes
            self.users = users
            self.movies = movies

            # Initialize both matrix from SVD
            numUsers = len(users)
            numMovies = len(movies)

            ## Users preferences: numUsers * numLatentFactors
            self.usersPreferences = np.full(
                shape=(numUsers, self.numLatentFactors),
                fill_value=self.initializationValue,
                dtype=float)

            ## Movies description: numMovies * numLatentFactors
            self.moviesPreferences = np.full(
                shape=(numMovies, self.numLatentFactors),
                fill_value=self.initializationValue,
                dtype=float
            )

            # Control initialization of the system
            self.initialized = True

    def predict(self, user, movie):
        """
        Predict rating to the item for the given user.
        :param user:
        :param movie:
        :return: predicted rating.
        """

        # Get user and movies indexes.
        userIndex = self.userIndexes[user]
        movieIndex = self.moviesIndexes[movie]

        # Extract arrays for user preferences and movie descriptions
        userPreferences = self.usersPreferences[userIndex, :]
        movieDescriptions = self.moviesPreferences[movieIndex, :]

        # Perform scalar product for bot vectors
        predictedRating = np.dot(userPreferences, movieDescriptions)

        # Return predicted value
        return predictedRating

    def train_system(self):
        """
        Method to train the SVD system using Stochastic Gradient Descent.
        """

        # For each feature
        for feature in range(self.numLatentFactors):

            # Initialize cache for this feature
            self.init_cache(feature)

            # If latent factors file exists, not to train.
            if os.path.isfile(cfg.matrix):
                self.load_data()
                return

            # Train during numEpochs iterations
            for epoch in range(self.numEpochs):
                print("Training epoch {} for feature {}".format(epoch + 1, feature + 1))

                # Get user and movie values from users preferences and movie descriptions for this feature
                userValue = self.usersPreferences[:, feature]
                movieValue = self.moviesPreferences[:, feature]

                # errors
                errors = []

                # Iterate throughout all ratings
                for (user, movie) in self.ratingsTuples:

                    # Retrieve user and movie indexes
                    userIndex = self.userIndexes[user]
                    movieIndex = self.moviesIndexes[movie]

                    # Predict
                    predictedRating = self.predict_precalculated(user, movie, feature)

                    # Calculate error
                    err = self.ratingsTuples[(user, movie)] - predictedRating
                    errors.append(err)

                    # Perform Gradient Descent
                    initialUserValue = userValue[userIndex] # To avoid that update on userValue modifies update on movieValue
                    userValue[userIndex] += self.learningRate * (err * movieValue[movieIndex] - self.regularizeParameter * userValue[userIndex])
                    movieValue[movieIndex] += self.learningRate * (err * initialUserValue - self.regularizeParameter * movieValue[movieIndex])

                print("Avg error: {}".format(np.mean(errors)))

    def store_data(self):
        data = (self.usersPreferences, self.moviesPreferences)

        with open(cfg.matrix, 'wb') as matrix:
            pickle.dump(data, matrix)

    def load_data(self):

        with open(cfg.matrix, 'rb') as matrix:
            data = pickle.load(matrix)

        self.usersPreferences = data[0]
        self.moviesPreferences = data[1]

    def init_cache(self, feature):
        """
        This method is to precalculate ratings matrix except one feature. This will make the calculations faster.
        :param user:
        :param movie:
        :return: predicted rating.
        """

        # Extract arrays for user preferences and movie descriptions
        userPreferencesModified = np.delete(self.usersPreferences, feature, axis=1)
        movieDescriptionsModified = np.delete(self.moviesPreferences, feature, axis=1)

        # Perform scalar product for both matrix
        precalculatedRatingMatrix = np.dot(userPreferencesModified, movieDescriptionsModified.T)

        # Store precalculated matrix
        self.cache = precalculatedRatingMatrix

    def predict_precalculated(self, user, movie, feature):
        """
        Accelerate predictions during training.
        :param user:
        :param movie:
        :param feature:
        :return:
        """

        # Get user and movies indexes.
        userIndex = self.userIndexes[user]
        movieIndex = self.moviesIndexes[movie]

        # Get precalculated value
        predictedRating = self.cache[userIndex, movieIndex]

        # Calculate contribution from this feature to this rating
        predictedRating += self.usersPreferences[userIndex, feature] * self.moviesPreferences[movieIndex, feature]

        return predictedRating

    def query(self, user_id, query_limit=10):
        not_seen = set(self.movies).difference(set([id_movie for (id_movie, rating) in self.users[user_id].get_ratings()]))

        ranking = []

        for movie in not_seen:
            rating = self.predict(user_id, movie)
            ranking.append((movie, rating))

        # Sort ranking

        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        ranking = ranking[:10]

        recommendation = ""
        for movie in ranking:
            recommendation += str(movie[0]) + ". " + self.movies[movie[0]].get_title() + ": " + str(movie[1]) + "\n"

        return recommendation
