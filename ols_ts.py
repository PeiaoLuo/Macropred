# %%
import pandas as pd
import numpy as np
import os.path as osp
import json
from general_modules.datainit import manage

# %%
model_name = "ols_ts"
targetname = "EXP"

# skip to not scale
scaler_map = {
    "dif": "robust",
    "pct": "skip",
    "val": "minmax",
}

# process_y means if you want to winsorize target, outlier_bd set the winsorization Q
args = {
    "testlen": 12,
    "process_y": True,
    "outlier_bd": 0.0,
}

# from when you want to slice the data
date_cut_off = "2012-07"

# multicol : (use: bool, disablepca: bool)
# seasonal : (use: bool, method: within ["label", "fourier"], on_y: bool) 
process_setting = {
    "multicol": (True, False),
    "seasonal": (False, 'label', False)
}

# want to use engineered data
use_engineered = True

# %%
process_info = pd.read_excel(f"tables/{targetname}/transformation.xlsx")

# %%
# target , features will be transformed to real value type (spot) by default, if do not want, set transform = False
target, feature_dict, freq_dict = manage.load_data(target=targetname, engineered=use_engineered)


# %%
data_ls = []
for k,v in feature_dict.items():
    if not isinstance(v, pd.DataFrame):
        feature_dict[k] = None
    else:
        data_ls += [v]
feature = pd.concat(data_ls, axis=1)

# for basic transformation
# which features are advance
lag_name_dict = dict(zip(process_info['name'].tolist(),process_info['advance'].tolist()))
# transform to what type for each feature
transform_dict = dict(zip(process_info['name'].tolist(),process_info[model_name].tolist()))

# for scaler
# which scaler to use based on the type of data
scaler_dict = {}

for name, t in transform_dict.items():
    if name in feature.columns:
        scalername = scaler_map[t]
        if scalername not in scaler_dict.keys():
            scaler_dict[scalername] = [name]
        else:
            scaler_dict[scalername] += [name]

# %%
args.update(
    {
        "transform_dict": transform_dict,
        "scaler_dict": scaler_dict,
        "lag_name_dict": lag_name_dict,
    }
)

# %%
from general_modules.data_prep.prepare import CommonPrep
from general_modules.data_prep.model_prep import ModelPrep
from general_modules.models.shallow import Models

prep = CommonPrep(target=target, feature=feature)
prep.cut_off(st=pd.to_datetime(date_cut_off))

# if args['testlen'] = 0, will get X_train as whole features, true_y as whole target, y_train and X_test will be None
# if want transformation, set args['transform'] with the dict got above, if not, set it to None, if want to change which to transform to what, go to transformation.xlsx
# others are alike, if use, then set args['WHATYOUWAN'] = already got above, if not, set it to None, if want change:
#       go to the top to change scaler_dict
#       go to the transformation.xlsx to change the lag_name_dict

# if want no split and do further process, you may stop here, with setting:
#   _, X, Y, _ = prep.process(args=args) with args['testlen']=0 with args you want (for task will train test split later, recommand set args['scaler_dict']=None, for the scaling
#   process will cause information leak)
y_train, X_train, true_y, X_test = prep.process(args=args)

# %%
model_prep = ModelPrep()
processor = model_prep.get_processor(modelname=model_name)
# if you want to change what exactly happen about the process, go to model_prep class to change

y_train,X_train,X_test = processor(y_train,X_train,X_test, rm_multicol=process_setting['multicol'], seasonal=process_setting['seasonal'])
logs = model_prep.logs


models = Models()
model = models.get_model(modelname=model_name)
model(X_train=X_train, X_test=X_test, y_train=y_train, true_y=true_y, targetname=targetname)


