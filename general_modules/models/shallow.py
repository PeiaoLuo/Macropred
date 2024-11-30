import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

class Models:

    def get_model(self, modelname:str):
        model = self.__getattribute__(modelname)
        return model

    def ols_ts(self, X_train, X_test, y_train, true_y, targetname, plot: bool=True) -> None:
        from general_modules.models.shallow_eval import ols_ts_eval
        
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        model = sm.OLS(y_train, X_train).fit()
        summary = model.summary()
        y_pred = model.predict(X_test)
        y_fit = model.fittedvalues
        y_scale = np.mean(true_y)
        val_loss = np.mean((true_y[-len(y_pred):-1].values - y_pred[:-1].values) ** 2) / y_scale**2
        
        combine = pd.concat([true_y, y_fit, y_pred], axis=1)
        combine.columns = ['y_true', 'y_fitted', 'y_predicted']
        
        directionratio, dratio_test, wholeloss = ols_ts_eval(combine)
        
        base_path = f"result/{targetname}/ols_ts"
        if not os.path.exists(path=base_path):
            os.mkdir(base_path)
        
        if plot:
            # plot the actual values
            plt.figure(figsize=(12, 8))

            # Plot the three lines: y_true, y_fitted, and y_predicted
            plt.plot(combine.index, combine['y_true'], label="y_true", color="blue")
            plt.plot(combine.index, combine['y_fitted'], label="y_fitted", linestyle="--", color="green")
            plt.plot(combine.index, combine['y_predicted'], label="y_predicted", linestyle="--", color="red")

            plt.title(f"{targetname}_value")
            plt.xlabel("Year")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(base_path, f"{targetname}_value.png"))
            
            plt.close()
            
            if isinstance(directionratio, pd.Series):
                plt.figure(figsize=(12, 8))

                # Plot the three lines: y_true, y_fitted, and y_predicted
                plt.plot(directionratio.index, directionratio.values, label="directionratio", color="blue")
                plt.title(f"{targetname}")
                plt.grid(True)
                plt.savefig(os.path.join(base_path, f"{targetname}_directionratio.png"))
                plt.close()
                
        with open(os.path.join(base_path, "model_summary.txt"), "w") as fp:
            fp.write(str(summary.as_text()))

        with open(os.path.join(base_path, "evaluation.txt"), "w") as fp:
            fp.write(f"Next period pred: {str(combine.iloc[-1,-1])} \n")
            fp.write(f"whole loss: {str(wholeloss)} \n")
            fp.write(f"val loss: {str(val_loss)} \n")
            fp.write(f"direction_ratio: {str(directionratio)} \n")
            fp.write(f"direction_ratio_test: {str(dratio_test)}")
        
        combine.to_csv(os.path.join(base_path, "src.csv"))
    
    
