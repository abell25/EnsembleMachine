__author__ = 'anthony bell'

import numpy as np
from scipy.stats import mode
from scipy.optimize import fmin
from time import time

from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.cross_validation import train_test_split

class Ensembler:
    '''
    class Ensembler: Sklearn wrapper for fitting and predicting with a group of classifiers.
    '''
    def __init__(self, clfs, class_probs=True, lower_is_better=True):
        self.clfs = clfs
        self.clf_fit = [False for clf in clfs]
        self.class_probs = class_probs
        self.lower_is_better = lower_is_better

    def fit(self, X, y, X_test=None, y_test=None, incremental_fit=False, scorer=None, use_weights=False, use_fmin=False, use_forwards=False, use_backwards=False, model_stacking_clf=None):  
        """
           fit model
           
           parameters:
           -----------
           X: training examples
           y: training labels
           X_test: required if using use_weights or use_fmin.  test examples used to calculate weights.
           y_test: required if using use_weights or use_fmin.  test labels used to calculate weights.
           incremental_fit: if new models have been added, setting this to true only trains the new models added.
           scorer: required if using use_weights or use_fmin.  scorer used to calculate weights.
           use_weights: weight classifiers based on scores.
           use_fmin: use optimization to calculate weights.
        """
        self.idxs = None
        self.model_stacking_clf = model_stacking_clf
        
        if use_weights:
            self.w = np.ones(len(self.clfs))
        else:
            self.w = None
        
        if X_test is not None:
            #running test predictions for running totals
            y_preds = np.zeros((X_test.shape[0], len(self.clfs)))
        
        
        for i in range(len(self.clfs)):
            t0 = time()
            clf, clf_fit = self.clfs[i], self.clf_fit[i]
            if not incremental_fit or not clf_fit:
                clf.fit(X, y)
                self.clf_fit[i] = True
            
            if scorer:
                y_preds[:,i] = clf.predict(X_test)
                
                if use_weights:
                    self.w[i] = scorer(y_test, y_preds[:,i]) + 1e-6
                    if self.lower_is_better:
                        self.w[i] = 1.0 / self.w[i] # take inverse of scores if lower is better
                             
                t_diff = (time() - t0)
                print('{0} time: {1}, score: {2}, running score: {3}'.format( i, t_diff , scorer(y_test, y_preds[:,i]), scorer(y_test, y_preds[:,:(i+1)].mean(axis=1))))
                print("model: {0}".format(str(clf)))
            
        self.clf_fit = [True for clf in self.clfs]
                             
        if use_weights:
            #normalize to equal N, so mean averages to 1
            self.w = self.w * ( (1/float(np.sum(self.w))) * float(len(self.clfs)) )

        if use_fmin:
            if self.w is None:
                raise Exception("use_weights=True is required for use_fmin")
            w_0 = self.w
            f = self.get_f(y_preds, y_test, scorer)
            w = fmin(f, w_0)
            self.w = w
            self.w = self.w * ( (1/float(np.sum(self.w))) * float(len(self.clfs)) )
            print("fmin_score: {0}, normal: {1}".format(scorer(y_test, (y_preds * self.w).mean(axis=1)), scorer(y_test, y_preds.mean(axis=1))))
           
        if use_backwards:
            print("use backwards ensemble selection")
            self.idxs = self.backwards(X_test, y_test, scorer)
    
        if use_forwards:
            print("using forwards ensemble selection")
            self.idxs = self.forwards(X_test, y_test, scorer)
        
        if model_stacking_clf:
            y_preds_train = self.predict(X_train, return_all_preds=True)
            y_preds_test  = self.predict(X_test, return_all_preds=True)
            self.model_stacking_clf.fit(y_preds_train, y_train)
            y_pred = self.model_stacking_clf.predict(y_preds_test)
            print("model stacking ensemble score: {0}".format(scorer(y_pred, y_test)))
            
    def get_f(self, y_preds, y, score):
        return lambda w: -score(y, np.mean((y_preds * w)*(1/np.sum(w)), axis=1))
            
    def predict(self, X, return_all_preds=False, use_w=True, use_model_stacking_clf=True, use_idxs=True):
        self.y_preds = np.zeros((X.shape[0], len(self.clfs)))
 
        if return_all_preds or self.idxs or self.model_stacking_clf:
            use_w = False
        
        for i in range(len(self.clfs)):
            w = 1.0 if self.w is None or not use_w else self.w[i]
            self.y_preds[:,i] = self.clfs[i].predict(X) * w
        
        if return_all_preds:
            return self.y_preds
        
        if self.idxs and use_idxs:
            return self.y_preds[:,self.idxs].mean(axis=1)
        
        if self.model_stacking_clf and use_model_stacking_clf:
            return self.model_stacking_clf.predict(self.y_preds)
    
        
        y_pred = self.y_preds.mean(axis=1)
        if self.class_probs:
            return y_pred
        else:
            return np.vectorize(lambda n: int(round(n)))(y_pred)
        
    def addModels(self, clfs):
        for clf in clfs:
            self.clfs.append(clf)
            self.clf_fit.append(False)
            
    def getAverage(self, method='mean', custom_func=None):
        '''
        Returns average values from ensemble.
        method: mean, median, mode, geometric. (default mean)
        custom_func: custom aggregation function.  method with be ignored
        '''
        if custom_func is not None:
            return np.apply_along_axis(custom_func, 1, self.y_preds)
        
        if method is 'mean':
            return self.y_preds.mean(axis=1)
        elif method is 'median':
            return np.median(self.y_preds, axis=1)
        elif method is 'mode':
            return mode(self.y_preds, 1).mode.flatten()
        elif method is 'geometric':
            return np.apply_along_axis(lambda r: np.product(r)**(1/float(len(r))), 1, self.y_preds)
        else:
            raise Exception('method %s is not valid!' % (method))
            
    def BlendClassifiers(self, X, y, clf=None, scorer=None):
        if clf is None:
            clf = LogisticRegression(class_weight='auto')
            
        y_preds = self.predict(X, return_all_preds=True)
        y_preds_train, y_preds_test, y_train, y_test = train_test_split(y_preds, y, train_size=0.8)
        clf.fit(y_preds_train, y_train)
        y_pred = clf.predict(y_preds_test)
        print("ensemble (with classifier) score: {0}".format(scorer(y_pred, y_test)))
        
        return clf.predict(y_preds)
       
    # Forwards

    def forwards(self, X, y, score, max_iters=50):
        y_preds = self.predict(X, return_all_preds=True)
        
        num_clfs = y_preds.shape[1]
        print("num clfs: {0}".format(num_clfs))
        
        all_idxs = list(range(num_clfs))
        idxs = []
        num_iters = max(max_iters, num_clfs)
        best_score = 0.0
        if self.lower_is_better:
            best_score = 99999999.0
        for iter_i in range(num_iters):
            best_idx = -1
            
            for i in all_idxs:
                s = score(y, y_preds[:,idxs + [i]].mean(axis=1))
                delta = s - best_score
                if self.lower_is_better: 
                    delta = -delta
                if delta > 0:
                    best_score, best_idx = s, i
                   
            if best_idx == -1:
                print("no clf improved performance!  quitting..")
                return idxs
                
            idxs += [best_idx]
            print("iter {0}/{1}: clf: {2}, score: {3}".format(iter_i, num_iters, best_idx, best_score))
            best_idx = -1
            
        return idxs
    
    # Backwards
    
    def backwards(self, X, y, score, max_iters=50):
        y_preds = self.predict(X, return_all_preds=True)
        
        num_clfs = y_preds.shape[1]
        idxs = set(range(num_clfs))
        num_iters = min(max_iters, num_clfs-1) #should have at least 1 left!
        best_score = 0.0
        if self.lower_is_better:
            best_score = 99999999.0    
        for iter_i in range(num_iters):
            best_idx = -1
            
            for i in idxs:
                s = score(y, y_preds[:,list(idxs - set([i]))].mean(axis=1))
                delta = s - best_score
                if self.lower_is_better: 
                    delta = -delta
                if delta > 0:
                    best_score, best_idx = s, i
                   
            if best_idx == -1:
                print("no clf is worsening performance!  quitting..")
                
            idxs -= set([best_idx])
            print("iter {0}/{1}: clf: {2}, score: {3}".format(iter_i, num_iters, best_idx, best_score))
            best_idx = -1
            
        return list(idxs)
    
    
    def clf_subset_predict(self, idxs, X):
        y_preds = self.predict(X, return_all_preds=True)
        res = y_preds[:,idxs].mean(axis=1)
        return res
