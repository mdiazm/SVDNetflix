"""
Main script of the system.

"""

from recommender_system import SVDNetflix
from gui import GUI

if __name__ == "__main__":

    # Test Content Based Recommender System computation
    recommender_system = SVDNetflix()
    recommender_system.initialize_system()
    recommender_system.train_system()
    recommender_system.store_data()
    # recommender_system.load_data()

    gui = GUI(system=recommender_system)
    gui.configure(title="SVD Recommender System")
    gui.show()

    exit(0)