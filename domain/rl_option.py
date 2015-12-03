__author__ = 'anthony bell'

class rlOption():
    def __init__(self, name, values_to_select, returned_values):
        self.name = name
        self.values_to_select = values_to_select
        self.returned_values = returned_values

    def show_available_options(self):
        return self.values_to_select

    def pick_option(self, selected_value):
        index = self.values_to_select.index(selected_value)
        return self.returned_values


