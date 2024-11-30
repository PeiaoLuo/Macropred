import pandas as pd
import numpy as np

def directionRatio2(x:pd.Series, y:pd.Series, method: tuple=None, already_diff:bool=False) -> float:
    assert len(x) == len(y)
    if len(x) == 0:
        return np.NAN
    
    if not already_diff:
        res = y.diff().values * x.diff().values
    else:
        res = y.values*x.values
    
    def direction(res):
        newres = np.nan_to_num(res, nan=0)
        return (newres > 0).sum() / (newres != 0).sum()    
    
    if not method:
        ratio = direction(res) 
        return ratio
    else:
        if method[0] == "culmulate":
            cul_res = []
            for i in range(len(res)-1):
                cul_res += [direction(res[:i+1])]
            cul_res = pd.Series(data=cul_res)
            cul_res.index = x.index[1:]
            return cul_res
        elif method[0] == "rolling":
            size = method[1]
            rolling_res = pd.Series(res).rolling(window=size).apply(direction)
            rolling_res.index = x.index
            return rolling_res
        
def loss(x,y,method):
    assert len(x) == len(y)
    if len(x) == 0:
        return np.NAN
    
    if method == "mse":
        scale = np.mean(y)
        return np.mean((x-y)**2) / scale**2

def ols_ts_eval(df):
    # cols need be ['y_true', 'y_fitted', 'y_predicted']
    reshaped_df = pd.concat([df['y_true'], df['y_fitted'].fillna(0) + df['y_predicted'].fillna(0)], axis=1).dropna()
    directionratio = directionRatio2(reshaped_df['y_true'], reshaped_df[0], method=("rolling",12), already_diff=True)
    directionratio_test = directionRatio2(df['y_true'].fillna(0), df['y_predicted'].fillna(0), method=None, already_diff=True)
    lossval = loss(reshaped_df['y_true'], reshaped_df[0], method="mse")
    
    return directionratio, directionratio_test, lossval 