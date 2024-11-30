from sklearn.compose import ColumnTransformer
import sklearn.preprocessing as pre
import pandas as pd
import numpy as np
from typing import Optional

class CommonPrep:
    def __init__(self, target:pd.Series, feature: pd.DataFrame, workflow_dict: Optional[dict] = None,) -> None:
        
        self.target = target
        self.scaled_traget = None
        self.feature = feature
        self.scaled_feature = None
        
        self.wf_dict = workflow_dict
    
    def cut_off(self, st=None, ed=None) -> None:
        if st == None and ed == None:
            return
        if st:
            self.target = self.target.loc[st:]
            self.feature = self.feature.loc[st:]
        if ed:
            self.target = self.target.loc[:ed]
            self.feature = self.feature.loc[:ed]
    
    #------------------------------------------------Frequency & Sampling ops----------------------------------------------------
    def getfreq(self, se1: pd.Series) -> str:
        freq_thresholds = {'365': 'Y', '85': 'Q', '29': 'M', '6': 'W'}
        se = se1.dropna().copy()
        mean_delta = np.mean([(se.index[i] - se.index[i - 1]).days for i in range(1, len(se.index) - 1)]) if len(se.index) > 1 else 0
        e = 'D'
        for threshold, freq in freq_thresholds.items():
            if mean_delta > int(threshold):
                e = freq
                break
            else: 
                continue
        return e
    
    def getfreq_dict(self, df:pd.DataFrame) -> dict:
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        freq_dict = {}
        for col in df.columns:
            freq_dict[col] = self.getfreq(df[col])

        return freq_dict
    
    def detect_freq(self, freq_dict:dict, targetname: str=None) -> dict:
        
        freq_map = {'Y': '365', 'Q': '085', 'M': '029', 'W': '006', 'D': '001'}
        reverse_map = {'365': 'Y', '085': 'Q', '029': 'M', '006': 'W', '001': 'D'}
        
        freq_ls = []
        ls_to_store = []
        for colname,freq in freq_dict.items():
            freq_ls += [freq_map[freq]]
            ls_to_store += freq
        
        # what to return
        freq_info = {
            "freqs": ls_to_store,
            "high": reverse_map[sorted(freq_ls)[0]],
            "low": reverse_map[sorted(freq_ls)[-1]],
            "same": 1 if len(set(freq_ls))==1 else 0,
            "targetok": None,
        }
        
        if targetname != None:
            freq_info['targetok'] = (freq_dict[targetname] == freq_info['low'])
        
        return freq_info
    
    def resample(self, tosample: pd.Series, freq: str) -> pd.Series: # downward sampling
        if freq.upper() not in ['W', 'D']:
            freq += 'E'
        ret = tosample.resample(rule=freq.upper()).mean()  
        return ret
    
    def resample_df(self, df:pd.DataFrame, freq:str, update=False) -> pd.DataFrame:
        # resample to same freq
        series_ls = []
        for col in df.columns:
            series_ls += [self.resample(df[col], freq)]
        resampled_df = pd.concat(series_ls, axis=1)
        if update:
            self.df = resampled_df
        return resampled_df
            
    #------------------------------------------------Frequency & Sampling ops----------------------------------------------------
    
    
    #--------------------------------------------------Workflow preparation------------------------------------------------------
    
    #--------------------------------------------------Workflow preparation------------------------------------------------------
    

    #-------------------------------------------------Model based preparation----------------------------------------------------
    
    def process(self, args):
        
        # scaler
        if args['process_y']:
            from scipy.stats import mstats
            # Winsorizing at the 1st and 99th percentiles
            bd = float(args["outlier_bd"])
            self.scaled_traget = pd.DataFrame(mstats.winsorize(self.target.values, limits=[bd,bd]), columns=self.target.columns, index=self.target.index)
        else:
            self.scaled_traget = self.target
            
        # apply scaler to each cols according to their transformed type
        if isinstance(args["scaler_dict"],dict):
            self.scale_feature(args["scaler_dict"], args['testlen'])
        else:
            self.scaled_feature = self.feature
        
        # resample into lowest freq --> transform into target type --> shift down not advanced features --> train test split
        y_train, X_train, true_y, X_test = self.basic_approach(testlen=args["testlen"], transform_dict=args["transform_dict"], lag_name_dict=args["lag_name_dict"])
            
        return y_train, X_train, true_y, X_test
    
    def basic_approach(self, testlen: int = 0, transform_dict: dict = None, lag_name_dict: dict = None):
        
        # concat target and feature
        rawdf = pd.concat([self.scaled_traget, self.scaled_feature], axis=1)
        
        # frequency info for rawdf
        freq_dict = self.getfreq_dict(rawdf)
        # get the lowest frequency in the rawdf and make sure it equals to the frequency of traget
        freq_info = self.detect_freq(freq_dict, targetname=self.target.columns[0])
        assert freq_info['targetok'] == True, ("Feature frequency lower than target, not supported.")
        
        # resample all columns into the lowest frequency
        aligned_df = self.resample_df(rawdf, freq_info['low'])
        true_y = self.resample_df(self.target, freq_info['low'])
        
        def transform(transform_dict, aligned_df):
            transformed_ls = []
            for col in aligned_df.columns:
                method = transform_dict[col]
                if method == "val":
                    transformed_ls += [aligned_df[col]]
                else:
                    transformed_ls += [self.__getattribute__(method)(aligned_df[col])]
            res = pd.concat(transformed_ls,axis=1)
            return res
        # if transformation info for cols are set, do transformation to cols
        if isinstance(transform_dict, dict):
            aligned_df = transform(transform_dict, aligned_df)
            if transform_dict[self.target.columns[0]] != 'val':
                true_y = self.__getattribute__(transform_dict[self.target.columns[0]])(true_y)
        
        # if which column is not advance feature, shift it down                
        if isinstance(lag_name_dict, dict):
            for col in aligned_df.columns:
                if lag_name_dict[col] == 0:
                    aligned_df.loc[:,col] = aligned_df[col].shift(1)

        tgname = self.target.columns[0]
        # train test split
        if testlen:
            train_part = aligned_df.iloc[:-testlen]
            test_part = aligned_df.iloc[-testlen:]
        else:
            train_part = aligned_df
            train_part = train_part.ffill()
            
            y_train = train_part[tgname]
            y_train = y_train.replace(np.inf, y_train.dropna(how='any').mean())
            y_train = y_train.replace(-np.inf, y_train.dropna(how='any').mean())
            
            train_part = train_part.drop(columns=tgname)
            train_part = train_part.replace(np.inf, 0)
            train_part = train_part.replace(-np.inf, 0)
            return y_train, train_part, true_y, None    
            
        train = train_part.ffill()
        # train = train.fillna(0)
        train = train.dropna(how='any')
        X_train = train.drop(columns=tgname)
        y_train = train[tgname]
        X_test = test_part.drop(columns=tgname).ffill()
        
        X_train = X_train.replace(np.inf, 0)
        X_train = X_train.replace(-np.inf, 0)
        X_test = X_test.replace(np.inf, 0)
        X_test = X_test.replace(-np.inf, 0)
        y_train = y_train.replace(np.inf, y_train.dropna(how='any').mean())
        y_train = y_train.replace(-np.inf, y_train.dropna(how='any').mean())
        
        return y_train, X_train, true_y, X_test 
        
    #-------------------------------------------------Model based preparation----------------------------------------------------

    #-------------------------------------------------Transform----------------------------------------------------
    def pct(self, se:pd.Series) -> pd.Series:
        # assert isinstance(se, pd.Series), (f"Some thing wrong in interation of columns, get {type(se)}")
        se = se.dropna(how="any")
        se = se.pct_change()
        # se.fillna(0)
        se.replace(np.inf, 0, inplace=True)
        se.replace(-np.inf, 0, inplace=True)
        return se
    
    def dif(self, se:pd.Series) -> pd.Series:
        se = se.dropna(how="any")
        se = se.diff(1)
        return se

    #-------------------------------------------------Transform----------------------------------------------------
    
    #-------------------------------------------------Scalers-----------------------------------------------------
    def get_single_scaler(self,scaler_name):
        if scaler_name == "minmax":
            scaler = pre.MinMaxScaler(feature_range=(0,1))
        elif scaler_name == "standard":
            scaler = pre.StandardScaler()
        elif scaler_name == "robust":
            scaler = pre.RobustScaler()
        elif scaler_name == "skip":
            scaler = None
        else:
            raise ValueError(f"No such scaler: {scaler_name} supported")
        return scaler
    
    def scale_feature(self, scaler_dict, testlen):
        
        if testlen == 0:
            train_part = self.feature
            test_part = self.feature.iloc[:0, :]
        else:
            train_part = self.feature.iloc[:-testlen, :]
            test_part = self.feature.iloc[-testlen:, :]
        
        train_ls = []
        test_ls = []
        
        for scalername,cols in scaler_dict.items():
            for col in cols:
                if col in self.feature.columns:
                    train_tp_df = pd.DataFrame(train_part.loc[:,col].dropna(how='any'))
                    test_tp_df = pd.DataFrame(test_part.loc[:,col].dropna(how='any'))
                    scaler = self.get_single_scaler(scalername)
                    if scaler:
                        scaler.fit(train_tp_df)
                        train_ls += [pd.DataFrame(scaler.transform(train_tp_df), index=train_tp_df.index, columns=[col])]
                        if test_tp_df.shape[0] == 0:
                            continue
                        else:
                            test_ls += [pd.DataFrame(scaler.transform(test_tp_df), index=test_tp_df.index, columns=[col])]
                    else:
                        train_ls += [train_tp_df]
                        if test_tp_df.shape[0] == 0:
                            continue
                        else:
                            test_ls += [test_tp_df]
        
        scaled_train = pd.concat(train_ls, axis=1)
        if test_ls:
            scaled_test = pd.concat(test_ls, axis=1)
        else:
            scaled_test = None
        self.scaled_feature = pd.concat([scaled_train, scaled_test],axis=0)
        
        return