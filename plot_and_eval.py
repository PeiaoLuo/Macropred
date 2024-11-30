# %%
import pandas as pd
import numpy as np
import os.path as osp
import json
from general_modules.datainit import manage
import matplotlib.pyplot as plt

# %%
targetname = "EXP"
modelname = "ols_ts"
use_engineered = True

# %%
target, feature_dict, freq_dict = manage.load_data(target=targetname, engineered=use_engineered)
predres = pd.read_csv(f"result/{targetname}/{modelname}/src.csv", index_col=0, parse_dates=[0])

# %%
df = pd.merge(left=target, right=predres, left_index=True,right_index=True, how='outer')

# %%
df[f'{df.columns[0]}_fitted'] = df['y_fitted'] + df[df.columns[0]].shift(1)
df[f'{df.columns[0]}_predicted'] = df['y_predicted'] + df[df.columns[0]].shift(1)

# %%
df = df.loc[df[f'{df.columns[0]}_fitted'].first_valid_index():]

# %%
plt.figure(figsize=(18,6))

plt.plot(df.index, df[df.columns[0]])
plt.plot(df.index, df[f'{df.columns[0]}_fitted'])
plt.plot(df.index, df[f'{df.columns[0]}_predicted'])
plt.title(f"True_val_{targetname}")
plt.savefig(f"result/{targetname}/{modelname}/value.png")
plt.close()

# %%
df[f'yoy_{df.columns[0]}'] = round((df[df.columns[0]] - df[df.columns[0]].shift(12)) / df[df.columns[0]].shift(12) * 100, ndigits=2)

df[f'combine_{df.columns[0]}'] = df[f'{df.columns[0]}_fitted'].fillna(0) + df[f'{df.columns[0]}_predicted'].fillna(0)

df[f'yoy_{df.columns[0]}_combine'] = round((df[f'combine_{df.columns[0]}'] - df[f'{df.columns[0]}'].shift(12)) / df[f'{df.columns[0]}'].shift(12) * 100, ndigits=2)

test_index = df[f'{df.columns[0]}_predicted'].first_valid_index()
df[f'yoy_{df.columns[0]}_predicted'] = df[f'yoy_{df.columns[0]}_combine'].loc[test_index:]
df[f'yoy_{df.columns[0]}_fitted'] = df[f'yoy_{df.columns[0]}_combine'].loc[:test_index]

# %%
plt.figure(figsize=(18,6))

plt.plot(df.index, df[f'yoy_{df.columns[0]}'])
plt.plot(df.index, df[f'yoy_{df.columns[0]}_fitted'])
plt.plot(df.index, df[f'yoy_{df.columns[0]}_predicted'], color='r')
plt.axhline(y=0, color='lightgreen')
plt.title(f"yoy_{targetname}")
plt.savefig(f"result/{targetname}/{modelname}/yoy.png")
plt.close()

# %%
from general_modules.models.shallow_eval import directionRatio2
predded = df[f'yoy_{df.columns[0]}_predicted']
fitted = df[f'yoy_{df.columns[0]}_fitted']
true = df[f'yoy_{df.columns[0]}']
rollingdr_fit = directionRatio2(fitted, true, method=("rolling", 12), already_diff=True)
dr_fit = directionRatio2(fitted, true, method=None, already_diff=True)
dr_pred = directionRatio2(predded, true, method=None, already_diff=True)

# %%
plt.figure(figsize=(18,6))
plt.plot(rollingdr_fit.index, rollingdr_fit)
plt.title("directionRatio")
plt.savefig(f"result/{targetname}/{modelname}/yoy_directionratio.png")
plt.close()

# %%
prediction = df.iloc[-1,:][f'yoy_{df.columns[0]}_predicted']
with open(f"result/{targetname}/{modelname}/yoy_evaluation.txt", 'w') as fp:
    fp.write(f"next period prediction: {prediction}\n")
    fp.write(f"fit direction ratio: {round(dr_fit, 4)}\n")
    fp.write(f"test direction ratio: {round(dr_pred, 4)}\n")


