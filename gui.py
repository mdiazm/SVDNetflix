"""
Script for Graphical User Interface
"""

from tkinter import *
from tkinter import scrolledtext

class GUI:
    """
    Class to handle graphical user interface
    """
    def __init__(self, system):
        self.window = Tk()
        self.system = system

    def configure(self, title):
        """
        Method to configure GUI
        """
        # Window title
        self.window.title(title)

        # Avoid window from RESIZING
        self.window.resizable(0, 0)

        # Labels and buttons
        self.user_id_label = Label(self.window, text="User id ")
        self.query_limit_label = Label(self.window, text="Limit of movies for this query")
        self.title_label = Label(self.window, text="Recommender System", font=("Arial Bold", 25))
        self.user_id_txt = Entry(self.window, width=5)
        self.query_limit_txt = Entry(self.window, width=5)
        self.query_button = Button(self.window, text="Get movies!", command=self.query)

        # Establish positions
        self.title_label.grid(column=0, row=0, sticky='we', columnspan=4, pady=20)
        self.user_id_label.grid(column=0, row=1, sticky='e')
        self.user_id_txt.grid(column=1, row=1, sticky='we', pady=10)
        self.user_id_txt.focus()
        self.user_id_txt.insert(END, '500')
        self.query_limit_label.grid(column=0, row=2, sticky='e')
        self.query_limit_txt.grid(column=1, row=2, sticky='we', pady=10)
        self.query_limit_txt.insert(END, '10')
        self.query_button.grid(column=2, row=1, sticky='wens', rowspan=2, columnspan=2, padx=5, pady=5)

        # TextArea to write down recommendations
        self.query_result = scrolledtext.ScrolledText(self.window)
        self.query_result.grid(column=0, row=3, columnspan=4, sticky='we')

    def show(self):
        """
        To show GUI
        """
        self.window.mainloop()

    def query(self):
        """
        This function will be executed when user press query button. Callback of the button
        """
        # Clear scrolled text
        self.query_result.delete(1.0, END)

        # Check textboxes are not none
        assert self.user_id_txt.get() != None
        assert self.query_limit_txt.get() != None

        # Get user id and query limit from textboxes
        user_id = int(self.user_id_txt.get())
        query_limit = int(self.query_limit_txt.get())

        # Make query to recommender system.
        recommendation = self.system.query(user_id, query_limit)

        # Add header
        recommendation = "Recommendations for user with id: {}\n************************************\n\n".format(user_id) + recommendation

        # Put text on ScrolledText area
        self.query_result.insert(INSERT, recommendation)
        