import pandas as pd
import os.path as osp
import json
from general_modules.datainit.maps_setting import category_map, equation_map
import copy
import re

CMAP = category_map
EMAP = equation_map

class ToVal():
    
    def __init__(self, df:pd.DataFrame, type_dict:dict) -> None:
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        self.df = df
        self.type_dict = type_dict
        
    def transform(self):
        new_col_ls = []
        for col in self.df.columns:
            temp_col = self.df[col].dropna(how="any")
            col_type = self.type_dict[col]
            if col_type == "val":
                new_col_ls += [temp_col]
            else:
                new_col_ls += [self.__getattribute__(col_type[:3])(temp_col, col_type)]
            
        transformed = pd.concat(new_col_ls,axis=1)
        
        return transformed
    
    def pct(self, se: pd.Series, formula):
        pattern = r"pct\+(\d+)\*(\d+)"
        matched = re.search(pattern=pattern, string=formula)
        add_element = int(matched.group(1))
        mul_element = int(matched.group(2))
        
        se = se / mul_element - add_element + 1
        se = se.cumprod()
        
        return se
        

def load_data(target, transform=True, engineered=True) -> dict[pd.DataFrame]:
    """load data according to feature dataframe, the data should already has column names align with name column in feature.xlsx"""
    
    if engineered:
        feature_df = pd.read_excel(osp.join("tables", target, "engineered.xlsx"))
        feature_df = feature_df[feature_df['use']==1]
    else:
        feature_df = pd.read_excel(osp.join("tables", target, "feature.xlsx"))
        feature_df = feature_df[feature_df['use']==1]
    
    # get target type and freq
    with open(osp.join("tables", target, "targetinfo.json"), "r") as fp:
        target_info_dict = json.load(fp)
    target_type_dict = {target_info_dict['name']: target_info_dict['type']}
    
    # frequency dict for target and feature
    freq_dict = dict(zip(feature_df['name'].tolist(), feature_df['freq'].tolist()))
    freq_dict.update({target_info_dict['name']: target_info_dict['freq']})
    
    # data storage path according to the macro target
    if engineered:
        dt_base_path = f"engineered_data/{target}"
    else:
        dt_base_path = f"data/{target}"
    
    # get feature categories given the target
    cmap = CMAP[target]
    
    # filter out features do not want to use
    feature_type_dict = dict(zip(feature_df['name'].tolist(), feature_df['type'].tolist()))
    
    # load data according to category and source info
    grouped_df = feature_df.groupby(["category","src"])
    
    # to store features
    df_dict = copy.deepcopy(cmap)
    equations = []
    
    for (category, src), df in grouped_df:
        
        # data path settings here
        if src == "wind":
            path = osp.join(dt_base_path, cmap[category], "wind.csv")
            usecols = df['name'].tolist()
            if not isinstance(usecols, list):
                print(1)
        elif src.startswith("self"):
            equations.append(src.split(':')[1])
            continue
        else:
            fname = df.loc[df.index[0], 'name'] + ".csv"
            path = osp.join(dt_base_path, cmap[category], fname)
        
        if src == "wind":
            data_to_add = pd.read_csv(path, index_col=0, parse_dates=[0], usecols=["指标ID"] + usecols)
        else:
            data_to_add = pd.read_csv(path, index_col=0, parse_dates=[0])

        try:
            if isinstance(df_dict[category],str):
                df_dict[category] = data_to_add
            elif isinstance(df_dict[category],pd.DataFrame):
                df_dict[category] = pd.concat([df_dict[category], data_to_add],axis=1)
            else:
                raise f"Something wrong in the df_dict. Got type {type(df_dict[category])} in the dict."
        except Exception as e:
            print(f"No such category (see the error message make sure the problem is supposed): {e}")
        
    
    for i in equations:
        try:
            exec(EMAP[target][i])
        except Exception as e:
            print("Error calculate according to function, check it (make sure the src to calculate this index is set to be used).")
    
    target_df_path = osp.join(dt_base_path, "target", "target.csv")
    target_df = pd.read_csv(target_df_path, parse_dates=[0], index_col=0)
    
    if transform:
        for name, df in df_dict.items():
            if isinstance(df, pd.DataFrame):
                instance = ToVal(df, feature_type_dict)
                result = instance.transform()
                df_dict[name] = result
        
        instance = ToVal(target_df, target_type_dict)
        target_df = instance.transform()
    
    return  target_df, df_dict, freq_dict