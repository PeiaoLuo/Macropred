import pandas as pd
import numpy as np
import minepy
from sklearn.metrics import mutual_info_score
from typing import Tuple

class Index():
    
    def __init__(self, data:pd.DataFrame, freq_dict: dict = None, targetname: list = None) -> None:
        self.targetname = targetname
        self.data = data
        self.columns = data.columns
        
        # for funcnames functions, input order does not matter
        # for order_funcnames functions, input order should be target first, feature second
        self.funcnames = ['corr', 'directionRatio2', "IGratio"]
        self.order_funcnames = ['MIC']
        
        # help compare str like frequency
        self.freq_map = {'Y': '365D', 'Q': '90D', 'M': '30D', 'W': '7D', 'D':'1D'}
        
        if freq_dict:
            self.freq_dict = freq_dict
        else:
            pass # import ... from ...
        
        self.static_indexes = None
    
    def get_weighted_scores(self, weights, index_names):
        if isinstance(self.static_indexes, pd.DataFrame):
            
            def func(df):
                res = 0
                for name, weight in weights.items():
                    value = df[df['index_name'] == name]['value']
                    if not value.empty:
                        res += abs(value).iloc[0] * weight      
                return res

            res = self.static_indexes[['feature', 'index_name', 'value']].groupby('feature').apply(func)
            
        else:
            self.calc_static_indexes(index_names=index_names)
            res = self.get_weighted_scores(weights,index_names)
            
        return res
    
    #START------------------------------------- combined index calc function
    def calc_static_indexes(self, index_names=["all"], overwrite: bool =False) -> pd.DataFrame:
        if self.static_indexes is not None:
            if not overwrite:
                return self.static_indexes
        
        res = {'target':[], 'feature':[], "freq":[], "length": [], 'index_name':[], "value":[]}
        
        # get all functions to calc indexes
        func_ls = []
        
        if index_names[0] == "all":
            for name in self.funcnames:
                func_ls += [getattr(self, name)]
            for name in self.order_funcnames:
                func_ls += [getattr(self, name)]
        else:
            for name in index_names:
                func_ls += [getattr(self, name)]
        
        # input target and feature series output index value
        def get_indexes(target, feature):
            indexes = []
            for func in func_ls:
                res = func(target, feature)
                indexes += [res]
            return indexes
        
        # if targets are given
        if self.targetname:
            index_num = len(func_ls)
            print(f"Targets given, calculating {index_num} indexes...")
            print(f"The indexes are: {index_names}")
            # split targets and features
            targets = self.data[self.targetname]
            features = self.data.drop(columns=self.targetname)
            
            # traverse targets
            for tgname in targets.columns:
                print(f"Calculating indexes for {tgname}...")
                res['target'] += [tgname]*len(features.columns)*index_num
                # traverse features
                for ftname in features.columns:    
                    res['feature'] += [ftname]*index_num
                    res['index_name'] += index_names
                    
                    # align target and feature, get the aligned freq and length
                    (tg, ft, freq, length) = self.align_and_combine(targets[tgname], features[ftname])
                    
                    index_vals = get_indexes(tg, ft)
                  
                    res['freq'] += [freq]*index_num
                    res['length'] += [length]*index_num
                    res['value'] += index_vals
                    
        # if no target is given, iteratively use every col as target
        else:
            return
        
        print("Index calculation finished.\n")
        self.static_indexes = pd.DataFrame(res)
        return
    #END--------------------------------------- combined index calc function
    
    
    #START------------------------------------- data downward alignment functions
    def resample(self, tosample: pd.Series, freq: str) -> pd.Series: # downward sampling
        if freq.upper() not in ['W', 'D']:
            freq += 'E'
        ret = tosample.resample(rule=freq.upper()).mean()  
        ret.dropna(how='any')
        return ret
    
    
    def align_and_combine(self, x:pd.Series, y:pd.Series) -> Tuple[pd.Series, pd.Series, str, int]: # make two series align in frequency
        xfreq = self.freq_dict[x.name]
        yfreq = self.freq_dict[y.name]
        if pd.to_timedelta(self.freq_map[xfreq]) > pd.to_timedelta(self.freq_map[yfreq]):
            frequency = xfreq
        else:
            frequency = yfreq
            
        x = self.resample(tosample = x, freq=frequency)
        y = self.resample(tosample = y, freq=frequency)
        
        temp = pd.concat([x,y],axis=1).dropna(how='any')
        x = temp[x.name]
        y = temp[y.name]
        
        if len(y) == 0:
            print(1)
        
        return (x, y, frequency, len(x))
    #END--------------------------------------- data downward alignment functions
        
        
    #START------------------------------------- static index calc functions
    def check(self, y, x): # check if two series align in length or have 0 length
        if len(y) == 0:
            return 0
        if len(y) != len(x):
            raise "Something wrong, target and feature length is not aligned, if you are trying to directly call the function, make sure the target and feature has gone through\
                 the align_and_combine function"
        return 1
    
    def corr(self, y:pd.Series, x:pd.Series) -> float: # Pearson correlation

        if self.check(y,x) == 0:
            return np.NAN
        
        return np.corrcoef(y.values, x.values)[0,1]
    
    
    def directionRatio2(self, x:pd.Series, y:pd.Series, already_dif = False, eval = False) -> float:
        
        if self.check(y,x) == 0:
            return np.NAN
        
        if not already_dif:
            res = y.diff().values * x.diff().values
        else:
            res = y.values * x.values
            
        res = np.nan_to_num(res, nan=0)
            
        ratio = (res > 0).sum() / (res != 0).sum()
        if not eval:
            ratio = abs(ratio - 0.5) / 0.5
        return ratio
        
    
    def IGratio(self, target:pd.Series, feature:pd.Series) -> float: # labelized IGRatio for series
        # 1 period diff, >0 ? 1 : 0
        def labelize(series:pd.Series):
            return (series.diff(1) > 0).astype(int)[1:]
        def calc_iv(series:pd.Series):
            prob = series.value_counts(normalize=True)
            return - sum(prob * np.log2(prob))
        tg = labelize(target)
        ft = labelize(feature)
        ig = mutual_info_score(tg.values, ft.values)
        iv = calc_iv(tg)

        igratio = ig/iv if iv != 0 else 0
        return igratio


    def MIC(self, x:pd.Series, y:pd.Series, m=True) -> float: # MIC score for series using minepy
        
        if self.check(y,x) == 0:
            return np.NAN
        
        mic = np.NAN
        
        if m:
            model = minepy.MINE(alpha=0.6, c=15)
            model.compute_score(x.values, y.values)
            mic = model.mic()
        else:
            pass
        
        return mic
    #END--------------------------------------- static index calc functions


