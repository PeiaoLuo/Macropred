model_name = "ols_ts"
targetname = "EXP"
scaler_map = {
    "dif": "robust",
    "pct": "skip" # it seems use scaler to pct value will make the performance much worse
}
args = {
    "testlen": 6,
    "process_y": True,
    "outlier_bd": 0.0,
}
date_cut_off = "2012-07"
process_setting = {
    "multicol": (True, False),
    "seasonal": (False, 'label', False)
}
use_engineered = True