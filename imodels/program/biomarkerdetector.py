import numpy as np
from lightgbm.sklearn import LGBMClassifier
from imblearn.over_sampling import ADASYN
from pyHSICLasso import HSICLasso
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit

class PHSICAdasynLGBM(BaseEstimator):
    """
    An estimator upsampling minority classes, finding a small set of 
    stable biomarkers, and fitting a gradient boosting model over them

    Parameters
    ----------
    n_features : int, optional (default=30)
        Max. number of biomarkers (important features) to be selected

    adasyn_neighbors : int, optional (default=10)
        K neighbors for ADASYN upsampling algorithm
        
    B : int, optional (default=20)
        Block size for Block HSIC Lasso
        
    M : int, optional (default=10)
        Max allowed permutations of samples for Block HSIC Lasso

    hsic_splits :  int, optional (default=5)
        number of folds for verifying feature stability

    feature_neighbor_threshold : float, optional (default=0.4)
        threshold for considering neighbors of important features in stability check
    """
    
    def __init__(self, n_features=30, adasyn_neighbors=10, B=20, M=10, hsic_splits=3, stability_minimum_across_splits=2, feature_neighbor_threshold=0.4):
        self.n_features=n_features
        self.adasyn_neighbors=adasyn_neighbors
        self.M=M
        self.B=B
        self.hsic_splits = hsic_splits
        self.neighbor_threshold = feature_neighbor_threshold
        self.stability_minimum_across_splits = stability_minimum_across_splits
    
    def fit(self, X, y):   
        if X.shape[1] > 10000:
            #clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1).fit(X,y)
            clf = LGBMClassifier(n_estimators=1000,n_jobs=-1).fit(X,y)
            ftimp = clf.feature_importances_
            relevant = np.where(ftimp>0)[0]
            print("relevant ft:",len(relevant),"/",X.shape[1])
        else:
            relevant = np.arange(X.shape[1])
        
        sss = StratifiedShuffleSplit(n_splits=self.hsic_splits,random_state=42)
        idxs = []
        hsics = []
        for train_index, test_index in list(sss.split(X, y)):
            hsic_lasso2 = HSICLasso()
            hsic_lasso2.input(X[:,relevant][train_index],y[train_index])
            hsic_lasso2.classification(self.n_features, B=self.B,M=self.M) #(self.n_features, B=self.B, M=self.M)
            hsics.append(hsic_lasso2)
            
            # not just best features - get their neighbors (similar features) too
            all_ft_idx = np.array(hsic_lasso2.get_index(),dtype=int).ravel()
            for i in range(len(all_ft_idx)):
                idx = np.array(hsic_lasso2.get_index_neighbors(feat_index=i, num_neighbors=10), dtype=int)
                score = np.array(hsic_lasso2.get_index_neighbors_score(feat_index=i, num_neighbors=10), dtype=int)
                idx = idx[np.where(score>self.neighbor_threshold)[0]]
                all_ft_idx = np.concatenate((all_ft_idx, idx))
            all_ft_idx = np.unique(all_ft_idx)
            
            idxs.append( relevant[all_ft_idx] )
            #if len(idxs) == 1:
            #    self.hsic_idx_ = idxs[0]
            #else:
            #    self.hsic_idx_ = np.intersect1d(idxs[-1], self.hsic_idx_)
        self.hsic_idx_ = []
        
        stability_concession = 0
        while len(self.hsic_idx_) == 0:
            featurecandidates = np.unique(np.concatenate(idxs))
            for candidate in featurecandidates:
                occurrences = np.sum([1 if candidate in idx else 0for idx in idxs])
                if occurrences > self.stability_minimum_across_splits - stability_concession:
                    self.hsic_idx_.append(candidate)
            if len(self.hsic_idx_) > 1:
                break
            else:
                # failed to find commonly occurring features - reduce threshold
                stability_concession += 1
        print("HSIC done.",len(self.hsic_idx_),"(out of ",len(featurecandidates)," candidates)")
        
        print("Upsampling with ADASYN... (features: "+str(len(self.hsic_idx_))+")")
        sm = ADASYN(sampling_strategy="minority", n_neighbors=self.adasyn_neighbors, n_jobs=-1)
        sX,sy = X[:,self.hsic_idx_],y
        if self.adasyn_neighbors > 0:
            try:
                sX, sy = sm.fit_resample(X[:,self.hsic_idx_],y)
                for i in range(len(np.unique(y)-1)):
                    sX, sy = sm.fit_resample(sX,sy)
            except:
                pass
            print("ADASYN done. Starting clf") 
        
        self.clf_ = LGBMClassifier(n_estimators=1000).fit(sX, sy)
        print("done")
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X[:,self.hsic_idx_])

    def predict(self, X):
        return self.clf_.predict(X[:,self.hsic_idx_])   