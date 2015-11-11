__author__ = 'anthony bell'

from train_test_data_loader import TrainTestDataLoader
from glob import glob
from os import path
import re
import pandas as pd

class ChalearnWrapper():
    def __init__(self, files_loc='.'):
        self.files_loc = files_loc

    def getDatasetFiles(self, dataset_loc, dataset_name, load_dataset=True):
        files = {
            'dataset_properties': '{0}_public.info'.format(dataset_name),
            'train_data': '{0}_train.data'.format(dataset_name),
            'test_data': '{0}_test.data'.format(dataset_name),
            'validation_data': '{0}_valid.data'.format(dataset_name),
            'train_labels': '{0}_train.solution'.format(dataset_name)
        }

        for k in files:
            files[k] = path.join(dataset_loc, files[k])

        if load_dataset:
            files['train_data'] = self.loadNoHeaderDataframe(files['train_data']),
            files['test_data'] = self.loadNoHeaderDataframe(files['test_data']),
            files['validation_data'] = self.loadNoHeaderDataframe(files['validation_data']),
            files['train_labels'] = self.loadNoHeaderDataframe(files['train_labels'])
            
        files['train_data'] = files['train_data'][0] if type(files['train_data']) is tuple else files['train_data']
        files['test_data'] = files['test_data'][0] if type(files['test_data']) is tuple else files['test_data']
        files['validation_data'] = files['validation_data'][0] if type(files['validation_data']) is tuple else files['validation_data']
        files['train_labels'] = files['train_labels'][0] if type(files['train_labels']) is tuple else files['train_labels']

        return files

    def getAvailableDatasets(self):
        path_glob = path.join(self.files_loc, 'round*/*')
        paths = glob(path_glob)
        return [x.split('/')[-1] for x in sorted(paths)]

    def getDataset(self, dataset_name, dataset_loc = None):
        files_loc = self.files_loc if dataset_loc == None else dataset_loc
        dataset_location = glob('{0}/*/{1}'.format(files_loc, dataset_name))[0]
        dataset_files = self.getDatasetFiles(dataset_location, dataset_name)
        dataset_files['dataset_properties'] = self.loadDatasetPropertiesDict(dataset_files['dataset_properties'])
        return dataset_files

    def loadDatasetPropertiesDict(self, dataset_properties_file):
        file_data = open(dataset_properties_file).read()
        lines = file_data.splitlines()
        tuples = [re.split(' *= *', line) for line in lines]
        d = {x[0]: x[1] for x in tuples}
        for key in d:
            val = d[key]
            if val[0] == val[-1] and val[0] in ['"', "'"]:
                d[key] = val[1:-1]
            else:
                d[key] = int(val)
                if key.startswith('has_') or key.startswith('is_'):
                    d[key] = True if d[key] == 1 else False

        return d

    def loadNoHeaderDataframe(self, filename, sep=' '):
        f = open(filename)
        df = pd.DataFrame(l.strip().split(sep) for l in f.readlines())
        return df

