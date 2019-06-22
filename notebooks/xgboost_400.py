
import os

if __name__ == "__main__":
    # NOTE: on posix systems, this *has* to be here and in the
    # `__name__ == "__main__"` clause to run XGBoost in parallel processes
    # using fork, if XGBoost was built with OpenMP support. Otherwise, if you
    # build XGBoost without OpenMP support, you can use fork, which is the
    # default backend for joblib, and omit this.
    try:
        from multiprocessing import set_start_method
    except ImportError:
        raise ImportError("Unable to import multiprocessing.set_start_method."
                          " This example only runs on Python 3.4")
    set_start_method("forkserver")

    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    import xgboost as xgb




    from sklearn.base import BaseEstimator
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor, XGBClassifier

    class XGBoostWithEarlyStop(BaseEstimator):
        def __init__(self, early_stopping_rounds=5, test_size=0.1, 
                     eval_metric='rmse', **estimator_params):
            self.early_stopping_rounds = early_stopping_rounds
            self.test_size = test_size
            self.eval_metric=eval_metric='rmse'        
            if self.estimator is not None:
                self.set_params(**estimator_params)

        def set_params(self, **params):
            return self.estimator.set_params(**params)

        def get_params(self, **params):
            return self.estimator.get_params()

        def fit(self, X, y):
            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size)
            self.estimator.fit(x_train, y_train, 
                               early_stopping_rounds=self.early_stopping_rounds, 
                               eval_metric=self.eval_metric, eval_set=[(x_val, y_val)])
            return self

        def predict(self, X):
            return self.estimator.predict(X)

    class XGBoostRegressorWithEarlyStop(XGBoostWithEarlyStop):
        def __init__(self, *args, **kwargs):
            self.estimator = XGBRegressor()
            super(XGBoostRegressorWithEarlyStop, self).__init__(*args, **kwargs)

    class XGBoostClassifierWithEarlyStop(XGBoostWithEarlyStop):
        def __init__(self, *args, **kwargs):
            self.estimator = XGBClassifier()
            super(XGBoostClassifierWithEarlyStop, self).__init__(*args, **kwargs)


    # Load data
    ABPRED_DIR = Path().cwd().parent
    DATA = ABPRED_DIR / "data"

    #dataframe final
    df_final = pd.read_csv(DATA/"../data/DF_features_400_2019.csv",index_col=0)
    # Quitar modelos por homologia deltraining set
    #df_final_onlyHM = df_final.loc[df_final.index.str.startswith("HM")]
    #df_final= df_final.loc[~df_final.index.str.startswith("HM")]

    index_ddg8 = (df_final['ddG(kcal/mol)']==8)
    df_final = df_final.loc[-index_ddg8]
    #testiar eliminando estructuras con ddg menor o igual a -4 kcal/mol , outliers
    index_ddg_4 =  (df_final['ddG(kcal/mol)'] <= -4)
    df_final = df_final.loc[-index_ddg_4]


    pdb_names = df_final.index
    features_names = df_final.drop('ddG(kcal/mol)',axis=1).columns

    # Data final
    X = df_final.drop('ddG(kcal/mol)',axis=1).astype(float)
    y = df_final['ddG(kcal/mol)']

    #Split data
    # split for final test
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,random_state=12)

    njob = 4
    os.environ["OMP_NUM_THREADS"] = str(njob)  # or to whatever you want
    
    xgb_model = XGBoostRegressorWithEarlyStop()
    param_grid = {'colsample_bytree': [0.3,0.6,0.7],
	'subsample': [0.5,0.6],
	'n_estimators':[50,100,150],
	'max_depth': [7,9,13,23],
    'gamma': [0.05,0.3,0.5,0.7,0.9],
    'learning_rate': [0.025],
    'min_child_weight':[1],
    'reg_lambda': [0.01,0.1,1]}

    grid = GridSearchCV(xgb_model, param_grid, verbose=5, n_jobs=njob,cv=10,scoring='neg_mean_absolute_error',return_train_score=True)
    grid.fit(X_train, y_train)


    print('CV test RMSE',np.sqrt(-grid.best_score_))
    print('CV train RMSE',np.sqrt(-grid.cv_results_['mean_train_score'].max()))

    print(grid.best_params_)
    y_test_pred = grid.best_estimator_.predict(X_test)
    y_train_pred = grid.best_estimator_.predict(X_train)

    print('Training score (r2): {}'.format(r2_score(y_train, y_train_pred)))
    print('Test score (r2): {}'.format(r2_score(y_test, y_test_pred)))

    print("\nRoot mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)))
    print("Root mean square error for train dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)))
    print("pearson corr: ",np.corrcoef(y_test_pred,y_test)[0][1])
