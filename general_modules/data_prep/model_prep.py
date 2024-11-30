import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

class ModelPrep:
    
    def __init__(self) -> None:
        self.logs = []
    
    def get_processor(self, modelname:str):
        processor = self.__getattribute__(modelname)
        return processor
    
    # -------------------------------------------processing methods--------------------------------------------
    def rm_multicollinearity(self, X_train, X_test, disable_pca=False):
        original_size = X_train.shape[1]
        
        X_origin = X_train.copy()
        test = 0
        if isinstance(X_test, pd.DataFrame):
            test = 1
        if X_train.isna().any().any():
            X_train = X_train.dropna(how='any')
        
        # Remove perfectly correlated features
        corr_threshold = 0.9
        corr_matrix = X_train.corr().abs()
        assert corr_matrix.isna().any().any() == False
        
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_triangle = pd.DataFrame(corr_matrix.values * mask, 
                                    index=corr_matrix.index, 
                                    columns=corr_matrix.columns)

        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
        
        if to_drop:
            X_train = X_train.drop(columns=to_drop)
            X_origin = X_origin.drop(columns=to_drop)
            if test:
                X_test = X_test.drop(columns=to_drop)
        
        usepca = 0
        if X_train.shape[1]/X_train.shape[0] > 0.5:
            warnings.warn(f"High dimention case, OLS not stable: sample_size/variable_num = {X_train.shape[1]}/{X_train.shape[0]}.")
            if not disable_pca:
                print("Will use PCA to remove feature size.")
                usepca=1
                # PCA
                from sklearn.decomposition import PCA

                # Fit PCA and retain components to 50% of the series length size
                pca = PCA(n_components=round(X_train.shape[0]*0.5))
                X_train_pca = pca.fit_transform(X_train)
                if test:
                    X_test_pca = pca.transform(X_test)
                
                X_train = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
                if test:
                    X_test = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

        vif_threshold = 10
        
        # VIF removal
        features_to_remove = []
        while True:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_train.columns
            try:
                vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
            except Exception as e:
                if "SVD did not converge" in e:
                    raise "SVD not converge for VIF for acceptable dimension data, need manual data quality check."
                    
            
            # Find features with high VIF
            high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
            
            if high_vif_features.empty:
                # No features with high VIF, break the loop
                break
            
            feature_to_remove = high_vif_features.loc[high_vif_features["VIF"].idxmax(), "feature"]
            features_to_remove += [feature_to_remove]
            
            # Remove the feature with the highest VIF
            if usepca:
                X_train = X_train.drop(columns=[feature_to_remove])
            else:
                X_train = X_train.drop(columns=[feature_to_remove])
                X_origin = X_origin.drop(columns=[feature_to_remove])
            if test:
                X_test = X_test.drop(columns=[feature_to_remove])
        
        if usepca:
            X_origin = X_train
        after_collinearity_size = X_train.shape[1]
        self.logs.append({"rm_muticol": f"Perfectly collinear cols: {to_drop}\nVIF large cols: {features_to_remove}\nFrom {original_size} to {after_collinearity_size}"\
            if usepca == 0 else "use PCA, no log"})
        
        return X_origin, X_test
    
    def seasonal_label(self, X_train, X_test):
        temp_df = pd.concat([X_train,X_test],axis=0)
        
        temp_df['month'] = temp_df.index.month
        temp_df = pd.get_dummies(temp_df, columns=['month'], dtype=np.float64)
        
        X_train = temp_df.iloc[:len(X_train),:]
        X_test = temp_df.iloc[len(X_train):]
        return X_train, X_test
        
    def seasonal_fourier(self, y_train, X_train, X_test):
        pass
    
    # --------------------------------------------models-------------------------------------------------
    def ols_ts(self, y_train, X_train, X_test, rm_multicol=(True, False), seasonal=(False, "label", False)):
        if seasonal[0]:
            if seasonal[1] == "label":
                X_train, X_test = self.seasonal_label(X_train, X_test)
            elif seasonal[1] == "fourier":
                if seasonal[3]:
                    pass
                else:
                    pass
        if rm_multicol[0]:
            X_train, X_test = self.rm_multicollinearity(X_train, X_test, disable_pca=rm_multicol[1])
        
        return y_train, X_train, X_test
    
    def transformer(self, y, x, rm_multicol):
        if rm_multicol[0]:
            x = self.rm_multicollinearity(x, None, disable_pca=rm_multicol[1])
    
    def lstm(self, y, x, ):
        pass
    
    def adaboost(self):
        pass