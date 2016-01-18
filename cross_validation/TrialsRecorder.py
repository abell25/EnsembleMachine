from pymongo import MongoClient
from bson.binary import Binary
import pickle

class TrialsRecorder:
    def __init__(self, dataset_name, host="localhost", port=27017):
        self.client = MongoClient(host, port)
        self.db_name = "Trials"
        self.dataset_name = dataset_name

        if not dataset_name:
            raise Exception("dataset_name is required to save trials to mongodb!")

        self.trials = self.client[self.db_name][dataset_name]

    def saveTrial(self, trial):
        if "y_oof" in trial:
            trial["y_oof"] = Binary(pickle.dumps(trial["y_oof"],  protocol=2), subtype=128)

        self.trials.insert_one(trial)


    def getAllTrials(self, query=None):
        for trial in self.trials.find(query):
            if "y_oof" in trial:
                trial["y_oof"] = pickle.loads(trial["y_oof"])
            yield trial

    def getAllTrialsWithScoreAtLeast(self, score):
        return self.getAllTrials({"score": {"$gte": score}})

    def getAllTrialsWithScoreAtMost(self, score):
        return self.getAllTrials({"score": {"$lte": score}})

    def clear_all_trials(self):
        self.client[self.db_name].drop_collection(self.dataset_name)

    @staticmethod
    def showTables(client=None):
        if not client:
            client = MongoClient()

        for db_name in client.database_names():
            print("DB: {0}".format(db_name))
            db = client[db_name]
            for tbl_name in db.collection_names():
                tbl = db[tbl_name]
                print("    collection: {0} (count: {1}, size: {2})".format(tbl_name, tbl.count(), db.command("collstats", tbl_name)['size']))
