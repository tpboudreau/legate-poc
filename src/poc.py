
import time
from pathlib import Path
#import pyarrow as pa
#import pyarrow.parquet as pq
import numpy as np
import cupynumeric as cn
import legate
from legate.core import get_legate_runtime as rt
import legateboost as lb
#import legate_dataframe as lf
from legate_dataframe.lib.parquet import parquet_read_array
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

def train():

    ds = Path('/mnt/lgt/tpboudreau/1MMx100.pqt') # (NEXT: increase to either 25MMx5000 or 10MMx10000)
    dg = '/mnt/lgt/tpboudreau/1MMx100.pqt/part.*.parquet'
    N = 1_000_000
    M = 100
    P = 25

    #ds = Path('/mnt/lgt/tpboudreau/5MMx1000.pqt')
    #dg = '/mnt/lgt/tpboudreau/5MMx1000.pqt/part.*.parquet'
    #N = 5_000_000
    #M = 1000
    #P = 25

    # Load from parquet to cupynumeric (or legate-dataframe)

    t0 = time.perf_counter()
    null_value = legate.core.Scalar(0, legate.core.float32)
    cs = [f'x{i+1}' for i in range(M)]
    Xr = parquet_read_array(dg, columns=cs, null_value=null_value, type=legate.core.float32)
    rt().issue_execution_fence(block=True)
    t1 = time.perf_counter()
    print(f"[read X] {t1-t0:.4f}")
    X = cn.asarray(Xr)
    rt().issue_execution_fence(block=True)
    t2 = time.perf_counter()
    print(f"[load X] {t2-t1:.4f}")
    #print(X.shape)
    #print(X)
    yr = parquet_read_array(dg, columns=['y'], null_value=null_value, type=legate.core.float32)
    rt().issue_execution_fence(block=True)
    t3 = time.perf_counter()
    print(f"[read y] {t3-t2:.4f}")
    y = cn.asarray(yr)
    rt().issue_execution_fence(block=True)
    t4 = time.perf_counter()
    print(f"[load y] {t4-t3:.4f}")
    #print(y.shape)
    #print(y)

    # Make some transformations (NEXT: legate dataframe might be a more natural alternative in some cases, e.g., with mixed data types)

    mu = cn.mean(y)
    sg = cn.sqrt(cn.var(y))
    y = (y - mu) / sg

    for i in range(M):
        # standardize all features
        mu = cn.mean(X[:, i])
        sg = cn.sqrt(cn.var(X[:, i]))
        X[:, i] = (X[:, i] - mu) / sg

        # (NEXT: perform other transformations when more interesting data is available: one-hot encoding, missing value imputation, etc)
        # ...

    rt().issue_execution_fence(block=True)
    t5 = time.perf_counter()
    print(f"[transformations] {t5-t4:.4f}")

    # Split the data into subpopulations

    R = cn.random.choice(3, N, p=[0.6, 0.2, 0.2])
    T = cn.argwhere(R==0).flatten()
    v = cn.argwhere(R==1).flatten()
    t = cn.argwhere(R==2).flatten()
    XT = X[T]
    Xv = X[v]
    Xt = X[t]
    yT = y[T]
    yv = y[v]
    yt = y[t]
    #print(XT.shape)
    #print(Xv.shape)
    #print(Xt.shape)
    #print(yT.shape)
    #print(yv.shape)
    #print(yt.shape)

    # Train the primary model (NEXT: add more sophistication here: categorical feature encoding, hyperparameter seaching, model ensembling)

    ev = {}
    #md = lb.LBRegressor(verbose=1, n_estimators=100, base_models=(lb.models.Tree(max_depth=20),)).fit(X, y)
    #md = lb.LBRegressor(verbose=1, n_estimators=100, base_models=(lb.models.Linear(), lb.models.Tree(max_depth=8),),).fit(X, y)
    #md = lb.LBRegressor(verbose=1, n_estimators=100, learning_rate=0.05, subsample=0.5, base_models=(lb.models.Tree(max_depth=8),)).fit(X, y)
    md = lb.LBRegressor(verbose=False, n_estimators=1000, learning_rate=0.05, subsample=0.5, base_models=(lb.models.Tree(max_depth=8),)).fit(XT, yT, eval_set=[(Xv, yv)], eval_result=ev)
    t6 = time.perf_counter()
    print(f"[primary training] {t6-t5:.4f}")

    # Obtain predictions on the holdout sample

    y_ = md.predict(Xt)
    rt().issue_execution_fence(block=True)
    t7 = time.perf_counter()
    print(f"[predictions] {t7-t6:.4f}")

    # Calculate some metrics

    me = mse(yt, y_)
    # [TODO] sklearn.metrics.r2_score() complains: TypeError: descriptor 'device' for 'numpy.ndarray' objects doesn't apply to a 'ndarray' object
    #r2 = r2s(y, y_)
    r2 = r2s(np.array(yt), np.array(y_))
    # [TODO] cupynumeric array has no setter for shape, cn.corrcoef complains of mismatched shapes for inputs
    #yt.shape = (yt.shape[0],)
    #cc = cn.corrcoef(yt, y_)
    at = np.array(yt)
    at.shape = (at.shape[0],)
    a_ = np.array(y_)
    cc = np.corrcoef(at, a_)
    t8 = time.perf_counter()
    print(f"[metrics] {t8-t7:.4f}")
    print(f"mse: {me:.8f}")
    print(f"r^2: {r2:.8f}")
    print(f"corr: {cc[0][1]:.8f}")

    # Distill the primary model if possible ...

    # ... into a model form with identical capacity
    ml = lb.LBRegressor(verbose=False, n_estimators=1000, learning_rate=0.05, base_models=(lb.models.Tree(max_depth=8),)).fit(Xt, y_)
    # ... into a model form with inferior capacity
    #ml = lb.LBRegressor(verbose=False, n_estimators=500, learning_rate=0.01, base_models=(lb.models.KRR(sigma=0.1),)).fit(Xt, y_)
    # ... into a model form with intermediate capacity
    #ml = lb.LBRegressor(verbose=False, n_estimators=500, learning_rate=0.05, base_models=(lb.models.Linear(),)).fit(Xt, y_)

    _y = ml.predict(Xt)
    me = mse(y_, _y)
    #r2 = r2s(y, _y) # see above
    r2 = r2s(np.array(y_), np.array(_y))
    #cc = cn.corrcoef(y_, _y) # see above
    a_ = np.array(y_)
    a_.shape = (a_.shape[0],)
    _a = np.array(_y)
    cc = np.corrcoef(a_, _a)
    print(f"mse: {me:.8f}")
    print(f"r^2: {r2:.8f}")
    print(f"corr: {cc[0][1]:.8f}")
    t9 = time.perf_counter()
    print(f"[total] {t9-t0:.4f}")

    return

if __name__ == '__main__':
    train()

