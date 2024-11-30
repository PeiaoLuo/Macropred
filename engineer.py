# %%
import pandas as pd
import numpy as np
import os.path as osp
import json
from general_modules.datainit import manage
from feature_engineer.index import Index

# %%
targetname = "EXP"
model_name = 'ols_ts'
feature_left = 40
args = {
    "testlen": 0,
    "process_y": True,
    "outlier_bd": 0.0,
}
index_names = ["MIC"]
weights = {
    "MIC": 1,
}

# %%
target, feature_dict, freq_dict = manage.load_data(target=targetname, engineered=False, transform=True)

# %%
process_info = pd.read_excel(f"tables/{targetname}/transformation.xlsx")
# which features are advance
lag_name_dict = dict(zip(process_info['name'].tolist(),process_info['advance'].tolist()))
# transform to what type for each feature
transform_dict = dict(zip(process_info['name'].tolist(),process_info[model_name].tolist()))
args.update(
    {
        "transform_dict": transform_dict,
        "scaler_dict": None,
        "lag_name_dict": lag_name_dict,
    }
)

# %%
data_ls = []
for k,v in feature_dict.items():
    if not isinstance(v, pd.DataFrame):
        feature_dict[k] = None
    else:
        data_ls += [v]
feature = pd.concat(data_ls, axis=1)

# %%
from general_modules.data_prep.prepare import CommonPrep
prep = CommonPrep(target=target, feature=feature)
_, x, y, _ = prep.process(args=args)

# %%
indexer = Index(data=pd.concat([x, y]), freq_dict=freq_dict, targetname=[y.columns[0]])
scores = indexer.get_weighted_scores(weights=weights, index_names=index_names)

# %%
scores.sort_values()

# %%
left_features = scores.sort_values().index.tolist()
left_features = left_features[-feature_left:]
if "IDR_CNY" in left_features:
    if "USD_IDR" not in left_features:
        left_features += ['USD_IDR']
    elif "USD_CNY" not in left_features:
        left_features += ['USD_CNY']

# %%
feature_df = pd.read_excel(f"tables/{targetname}/feature.xlsx")
feature_df.loc[~feature_df['name'].isin(left_features), 'use']=0
feature_df.to_excel(f"tables/{targetname}/engineered.xlsx")


