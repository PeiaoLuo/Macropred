category_map = {
    "EXP": {
        "AGG": "aggregate",
        "STR": "structure",
        "RPL": "replacement",
        "PRD": "product",
        "EXT": "extra"
    },
    "IMP": {
        "AGG": "aggregate",
        "STR": "structure",
        "RPL": "replacement",
        "PRD": "product",
        "EXT": "extra"
    },
    "M2": {
        "MON": "Monetary",
        "ECO": "Economy",
        "COM": "Commercial_Bank",
        "FIM": "Financial_Market",
        "FCB": "Family_and_Corporate_Behavior",
        "EXT": "External_Effect",
        "WLD": "International_Index"
    },
    "SF": {
        
    }
}

# designed for where src == "self:x", where x is the key
# if new derived features, add the expression here (for self:n as scr, add 'n': "equation" into map)
# standard format:
# df_dict[CATECORYNAME] -> sub dataframe of this category, you can apply any method, the equation will be excuted by exec()
equation_map = {
    "EXP": {
        '1': "df_dict['AGG']['IDR_CNY'] = df_dict['AGG']['USD_CNY'] / df_dict['AGG']['USD_IDR']",
    },
    "IMP": {
        '1': "df_dict['AGG']['IDR_CNY'] = df_dict['AGG']['USD_CNY'] / df_dict['AGG']['USD_IDR']",
    },
    "M2": {
        
    },
    "SF": {
        
    }
}