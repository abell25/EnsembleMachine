from unittest import TestCase
from model_library import ModelLibrary

__author__ = 'anthony bell'


class TestModelLibrary(TestCase):
  def setUp(self):
    pass

  def test_generateModel(self):
    model_library = ModelLibrary('classification', 'small')
    model = model_library.generateModel(model_selection_method='random', parameter_selection_method='random')


  def test_pickModel(self):
    model_library = ModelLibrary('classification', 'small')
    available_model_names = model_library.model_libraries['small classification'].keys()

    model_name = model_library.pickModel(model_selection_method='random')

    self.assertTrue(model_name in available_model_names)


  def test_pickParameters(self):
    model_library = ModelLibrary('classification', 'small')
    model_name = model_library.model_libraries['small classification'].keys()[0]
    uninitialized_model = model_library.model_libraries['small classification'][model_name]
    ModelClass, parameter_ranges = uninitialized_model['model'], uninitialized_model['parameter_ranges']

    chosen_parameters = model_library.pickParameters(parameter_ranges, parameter_selection_method='random')

    for param in parameter_ranges:
      self.assertTrue(chosen_parameters[param] in parameter_ranges[param])
