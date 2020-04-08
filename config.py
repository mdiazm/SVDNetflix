"""
Configuration of the system.

Paths, parameters and somewhat values.
"""

import os

# Directory of data files
DATA_PATH = "./data/"

# Users data
users = os.path.join(DATA_PATH, "users.csv")

# Movies data
movies = os.path.join(DATA_PATH, "movie-titles.csv")

# Movies tags
movies_tags = os.path.join(DATA_PATH, "movie-tags.csv")

# Ratings of the movies by the users
ratings = os.path.join(DATA_PATH, "ratings.csv")

# Serialized object for avoiding realoading csv files and parsing
serialized = os.path.join(DATA_PATH, "data.pickle")

# Serialized object for avoiding recalculating similarities between movies
similarities = os.path.join(DATA_PATH, "similarities.pickle")