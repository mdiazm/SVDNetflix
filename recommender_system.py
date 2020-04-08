"""
Script to hold code to construct the content-based recommender system.
"""

from data import Reader
from chrono import Timer
import numpy as np
import pickle
import config as cfg

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
    regularizeParameter = 0.02
    numEpochs = 120

    def __init__(self):
        self.tag_movie = {}
        self.initialized = False
        self.movie_similarities = {}

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

            # Train during numEpochs iterations
            for epoch in range(self.numEpochs):
                print("Training epoch {} for feature {}".format(epoch + 1, feature + 1))

                # Get user and movie values from users preferences and movie descriptions for this feature
                userValue = self.usersPreferences[:, feature]
                movieValue = self.moviesPreferences[:, feature]

                # Iterate throughout all ratings
                for (user, movie) in self.ratingsTuples:

                    # Retrieve user and movie indexes
                    userIndex = self.userIndexes[user]
                    movieIndex = self.moviesIndexes[movie]

                    # Predict
                    predictedRating = self.predict(user, movie)

                    # Calculate error
                    err = self.ratingsTuples[(user, movie)] - predictedRating

                    # Perform Gradient Descent
                    userValue[userIndex] += self.learningRate * (err * movieValue[movieIndex] - self.regularizeParameter * userValue[userIndex])
                    movieValue[movieIndex] += self.learningRate * (err * userValue[userIndex] - self.regularizeParameter * movieValue[movieIndex])

    def store_data(self):
        data = (self.usersPreferences, self.moviesPreferences)

        with open(cfg.matrix, 'wb') as matrix:
            pickle.dump(data, matrix)

    def load_data(self):

        with open(cfg.matrix, 'rb') as matrix:
            data = pickle.load(matrix)

        return data[0], data[1]