__author__ = 'anthony bell'

from sklearn.ensemble import RandomForestClassifer, RandomForestRegression

""" Feature selection:
    options: Forward/Backwards Selection
             PCA
             feature importances
             random subset
"""

class FeatureSelection():
    def __init__(self, lower_is_better = True, method=None, X=None, y=None):
        self.lower_is_better = lower_is_better
        self.method = method
        self.X = X
        self.y = y
        self.idxs = []
    
    def getTransformsList(self, method='all'):
        return {'all': allSelection,
                'forwards': forwardsSelection,
                'backwards': backwardsSelection,
                'importances': featureImportancesSelection,
                'random': randomSubsetSelection
               }
    
    def transform(self):
        pass
    
    def allSelection(self, X):
        """ all selection:
                 returns all features
        """
        return X
    
    def forwardsSelection(self, clf, X, y):
        """ forwards selection:
                add features 1-by-1 until score no longer improves

        """
        pass
    
    def backwardsSelection(self, clf, X, y):
        """ backwards selection:
                remove features 1-by-1 until score no longer improves
        """
        pass
    
    def featureImportancesSelection(self, X, y):
        """feature Importances selection:
                  uses rf/xgb feature importances to filter useless features
        """
        pass
    
    def randomSubsetSelection(self, X, y):
        """ random Subset selection
                  returns a random subset of the variables
        """
        pass
    
    def pcaSelection(self, X, y):
        """ pca selection
                returns top N
        """
    
def forwards(X, y, score, lower_is_better=False, clf_names=None):    
    num_clfs = X.shape[1]
    print "num features:", num_clfs
    
    clf_names = [str(n) for n in range(num_clfs)] if clf_names is None else clf_names
    
    all_idxs = list(range(num_clfs))
    idxs = [] if idxs is None else idxs
    num_iters = num_clfs
    best_score = 0.0
    for iter_i in range(num_iters):
        best_idx = -1
        
        for i in all_idxs:
            s = score(y, X[:,idxs + [i]].mean(axis=1))
            delta = s - best_score
            if lower_is_better: 
                delta = -delta
            if delta > 0:
                best_score, best_idx = s, i
               
        if best_idx == -1:
            print "no clf improved performance!  quitting.."
            return idxs
            
        idxs += [best_idx]
        print "iter %d/%d: clf: %d (%s), score: %.8f" % (iter_i, num_iters, best_idx, clf_names[best_idx], best_score)
        best_idx = -1
        
    return idxs

# Backwards

def backwards(X, y, score, lower_is_better=False, clf_names=None):    
    num_clfs = X.shape[1]
    idxs = set(range(num_clfs))
    num_iters = min(max_iters, num_clfs-1) #should have at least 1 left!
    best_score = 0.0
    for iter_i in range(num_iters):
        best_idx = -1
        
        for i in idxs:
            s = score(y, X[:,list(idxs - set([i]))].mean(axis=1))
            delta = s - best_score
            if lower_is_better: 
                delta = -delta
            if delta > 0:
                best_score, best_idx = s, i
               
        if best_idx == -1:
            print "no clf is worsening performance!  quitting.."
            
        idxs -= set([best_idx])
        print "iter %d/%d: clf: %d (%s), score: %.8f" % (iter_i, num_iters, best_idx, clf_names[best_idx], best_score)
        best_idx = -1
        
    return list(idxs)


def clf_subset_predict(X, idxs):
    return X[:,idxs].mean(axis=1)