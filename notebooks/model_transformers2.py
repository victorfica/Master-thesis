from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
import numpy as np

class FeaturesInteractions(BaseEstimator, TransformerMixin):
    
    #Class constructor method that takes ..
    def __init__(self, features1, feature2 ):
        self.features1 = features1
        self.feature2 = feature2
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        
        return self
    
    def transform(self, X , y=None ):
        
        X_interactions = np.multiply(X[:,self.features1],X[:,self.feature2:self.feature2+1])
        #X_interactions.columns = X_interactions.columns.values+'/{}'.format(self.interaction2)
                
        X = np.hstack([X,X_interactions])
        
        return X
    
class SkewTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
                
    #Return self nothing else to do here
    def fit(self, X, y = None  ):
        index_col = np.arange(X.shape[1])
        
        skewes_ = stats.skew(X)
        skew_mask = np.abs(skewes_) > self.threshold
        self.skew_cols = index_col[skew_mask]
        
        #self.t = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
        self.t = QuantileTransformer(output_distribution="normal",random_state=13)
        self.t.fit(X[:,self.skew_cols])
        
        return self
    
    def transform(self, X, y=None):
        
        X[:,self.skew_cols] = self.t.transform(X[:,self.skew_cols])
        
        return X
        
class ZeroThreshold(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=90.):
        self.threshold = threshold
    
    
    def fit(self, X, y = None  ):
        
        self.overfit = []
        for i,col in enumerate(X.T):
            zero_els = np.count_nonzero(col==0)
            if (zero_els / col.shape[0]) * 100 >self.threshold:
                self.overfit.append(i)
        
        return self
    
    def transform(self, X, y=None):
        
        return np.delete(X,self.overfit,axis=1)