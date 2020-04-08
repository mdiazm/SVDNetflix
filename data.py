"""
Class to hold objects that read data from files.
"""

import pandas as pd
import config as cfg
import pickle
import os

class User:
    """
    Class to hold user data

    """
    def __init__(self, id_user, description, ratings):
        """
        Constructor of user class.

        :param id_user: id of the user which is being stored
        :param description: of the user. Could be unique
        :param ratings: pairs of (movie, score) given from the user to movies.
        """
        self.id_user = id_user
        self.description = description
        self.ratings = ratings
        self.seen = {}

        # Store key of the already seen films
        for rating in ratings:
            self.seen[rating[0]] = True

    def to_string(self):
        """
        User description formatted as string
        """
        return "User: {} Description: {} Ratings: {}".format(self.id_user, self.description, self.ratings)

    def get_user_id(self):
        """
        Getter of the user id.
        """
        return self.id_user

    def get_ratings(self):
        """
        Getter of ratings list for this user.
        """
        return self.ratings

    def check_movie_seen(self, id_movie):
        """
        Method to check if a movie have been already seen for the given user.
        :param: id_movie: id of the movie to check

        return: boolean indicating if user saw the film
        """

        if id_movie in self.seen:
            return True

        return False

class Movie:
    """
    Class to hold movie data
    """

    def __init__(self, id_movie, title, tags, ratings):
        """
        Constructor of the movie class

        :param id_movie: of the movie
        :param title: of the movie
        :param tags: associated with a movie (to construct a vector space model in later stages)
        """

        self.id_movie = id_movie
        self.title = title
        self.tags = tags
        self.ratings = ratings

    def get_tags(self):
        """
        Getter of movie tags
        :return: movie tags
        """

        return self.tags

    def get_movie_id(self):
        """
        Getter of movie id
        :return: movie id
        """

        return self.id_movie

    def get_title(self):
        """
        Getter of movie title
        :return: movie title
        """

        return self.title

    def get_ratings(self):
        """
        Getter of ratings list for this movie. Pair (user, rating)
        """
        return self.ratings

class Reader:
    """
    Class to read data from files and store them in specific data structures.
    """

    def __init__(self):

        """
        Constructor: read every csv file using pandas.
        """

        print("Reading files from csv...")

        self.users = pd.read_csv(cfg.users, header=None)
        self.movies = pd.read_csv(cfg.movies, header=None)
        self.movies_tags = pd.read_csv(cfg.movies_tags, header=None, encoding="ISO-8859-1")
        self.ratings = pd.read_csv(cfg.ratings, header=None)

    def get_users(self):
        """
        Get users with their ratings of the films

        :return: dictionary key=user_id value = User object with every user's data.
        """
        users = {}

        for index, row in self.users.iterrows():

            user_id = int(row[0])
            description = row[1]
            user_ratings = []

            ratings = self.ratings.loc[self.ratings[0] == user_id]

            # Get user ratings
            for index_ratings, row_rating in ratings.iterrows():

                user_ratings.append((row_rating[1], float(row_rating[2])))  # (movie, score)

            # Append user object in dictionary
            users[user_id] = User(id_user=user_id, description=description, ratings=user_ratings)

        return users

    def get_movies(self):
        """
        Get movies with their tags

        :return: dictionary key=id_movie value = Movie object
        """

        movies = {}

        for index, movie in self.movies.iterrows():

            id_movie = int(movie[0])
            title = movie[1]
            movie_tags = []
            movie_ratings = []

            # Get tags for each movie
            tags = self.movies_tags.loc[self.movies_tags[0] == id_movie]

            for index_movie, tag in tags.iterrows():

                movie_tags.append(tag[1])

            # Get ratings for each film
            ratings = self.ratings.loc[self.ratings[1] == id_movie]

            # Get user ratings
            for index_ratings, row_rating in ratings.iterrows():
                movie_ratings.append((row_rating[0], float(row_rating[2])))  # (id_user, score)

            # Create Movie object
            movie_object = Movie(id_movie=id_movie, title=title, tags=movie_tags, ratings=movie_ratings)

            # Append movie_object in dictionary
            movies[id_movie] = movie_object

        return movies

    def write_serialized(self):
        """
        Write serialized object in a file

        :return: data which has been read in previous stages
        """

        # If file is yet created, return data and do not create it again
        if os.path.isfile(cfg.serialized):
            return None

        print("Creating users array...")
        users = self.get_users()

        print("Creating movies array...")
        movies = self.get_movies()

        data = (users, movies)

        # Save data as serialized object
        with open(cfg.serialized, 'wb') as serialized:
            print("Storing data as serialized object...")
            pickle.dump(data, serialized)

        # Return data whenever this function is called
        return data

    def load_serialized(self):
        """
        Load file from pickle or create it if not exists
        :return: pair of users, movies
        """
        if not os.path.isfile(cfg.serialized):
            serialized = self.write_serialized()
        else:
            print("Serialized object exists. Reading from disk...")
            with open(cfg.serialized, 'rb') as file:
                serialized = pickle.load(file)

        return serialized[0], serialized[1]  # users, movies

    def write_similarities(self, data):
        """
        Write similarities data to file
        :param data: similarities data
        """
        # If file is yet created, return data and do not create it again
        if os.path.isfile(cfg.similarities):
            return None

        with open(cfg.similarities, 'wb') as similarities:
            print("Storing data as serialized object...")
            pickle.dump(data, similarities)


    def load_similarities(self):
        """
        Load similarities pickle data
        :return: similiarities between pairs of movies
        """
        if not os.path.isfile(cfg.similarities):
            return None
        else:
            print("Serialized object exists. Reading from disk...")
            with open(cfg.similarities, 'rb') as file:
                data = pickle.load(file)

        return data