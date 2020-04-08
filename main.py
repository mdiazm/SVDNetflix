"""
Main script of the system.

"""

from recommender_system import CollaborativeFilteringRS
from gui import GUI

if __name__ == "__main__":

    # Test Content Based Recommender System computation
    recommender_system = CollaborativeFilteringRS()
    recommender_system.initialize_system()

    gui = GUI(system=recommender_system)
    gui.configure(title="Collaborative-filtering Recommender System")
    gui.show()

    exit(0)