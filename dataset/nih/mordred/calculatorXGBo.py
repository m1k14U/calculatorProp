# Dataset: NIH
# Prediction properties algorithm
# Model: Extreme Gradient Boosting 
# Descriptor: mordred

import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred.error import Error, Missing
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class ChemCalculatorXGRBoost:
    def __init__(self, csv_file):
        self.valid_smiles = []
        self.X = []
        self.features_names = ""
        self.calculator = Calculator(descriptors, ignore_3D = True)
        self.data = pd.read_csv(csv_file)
        self.model = XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror"
                    )

    def show_data(self):
        # Show the dataframe
        print(self.data)

    def select_data(self,number_of_data,feature:str,property:str):
        # Select the number of data and the features (exm. smiles and property)
        self.data = self.data[1:number_of_data]
        self.smiles_data = self.data[feature]
        self.property_data = self.data[property]

    def smiles_to_matrix_descriptors(self):
        # Pass the smiles to descriptors matrix
        if self.smiles_data is not None:
            self.list_smile = [Chem.MolFromSmiles(smi) for smi in self.smiles_data]
            self.descriptor = self.calculator.pandas(self.list_smile, nproc=1)
            self.descrip_cl = self.descriptor.applymap(lambda x: np.nan if isinstance(x, (Error, Missing)) else x)
            self.descrip_cl = self.descrip_cl.dropna(axis=1, how="all")
            self.descrip_cl = self.descrip_cl.dropna(axis=1, thresh=len(self.descriptor)*0.8)
            self.features_names = self.descrip_cl.columns.tolist()
            self.X = self.descrip_cl.to_numpy()
        else:
            print("Could not be possible to create a matrix. Make sure you select the molecules features with select_data()")

    def input_smiles_to_matrix(self,lisml):
        # The NEW smiles list to predict on a descriptors matrix
        self.X_new = [Chem.MolFromSmiles(smi) for smi in lisml]
        self.descriptor = self.calculator.pandas(self.X_new, nproc=1)
        self.descrip_cl = self.descriptor.applymap(lambda x: np.nan if isinstance(x, (Error, Missing)) else x)
        self.descrip_cl = self.descrip_cl.dropna(axis=1, how="all")
        self.descrip_cl = self.descrip_cl.dropna(axis=1, thresh=len(self.descriptor)*0.8)
        self.descrip_cl = self.descrip_cl.reindex(columns=self.features_names)
        self.X = self.descrip_cl.to_numpy()
        return self.X

    def get_shape_matrix_descriptors(self):
        # Show the shape of the 
        print(self.X.shape)

    def prepare_data(self):
        # Select the percentage of train data and define the X_train, y_train, X_test and y_test
        if self.X.any() and self.property_data.any():
            self.y = self.property_data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            return self.X_train, self.X_test, self.y_train, self.y_test
        
    def get_realvalue(self):
        return self.y_test
    
    def get_predictions(self):
        return self.y_pred

    def train(self):
        # Train the model (X_train, y_train)
        if self.X_train.any() and self.y_train.any() is not None:
            self.model.fit(self.X_train,self.y_train)

            return self.model
    
    def prediction(self):
        # Make a prediction with (X_test) and obtain y_pred 
        if self.X_test.any() and self.y_test.any() is not None:
            self.y_pred = self.model.predict(self.X_test)
            print("MAE: ",mean_absolute_error(self.y_test, self.y_pred))
            print("R2: ", r2_score(self.y_test, self.y_pred))
            
            return self.model
        else:
            print("No data for test!")
    
    def input_prediction(self,smileList):
        # Make a prediction with the NEW smiles list (X_new)
        self.input_smiles_to_matrix(smileList)
        self.y_new = self.model.predict(self.X)
        print(self.y_new)

        return self.y_new

    def plot_regression(self, y_true, y_pred, ideal_line=True, cmap="viridis", color=None,
                        s=30, alpha=0.9, ax=None, xlabel="Actual", ylabel="Predicted",
                        title=None, equal_aspect=True, show_stats=True, stats_fmt="{name} = {value:.3f}"):

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape after ravel().")

        # Drop non-finite rows
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[m]
        y_pred = y_pred[m]

        # Create axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
        else:
            fig = ax.figure

        if color is not None:
            sc = ax.scatter(y_true, y_pred, color=color, s=s, alpha=alpha, edgecolor="none", zorder=3)
        else:
            sc = ax.scatter(y_true, y_pred, cmap=cmap, s=s, alpha=alpha, edgecolor="none", zorder=3)

        # Reference line y=x
        if ideal_line:
            lo = np.nanmin([y_true.min(), y_pred.min()])
            hi = np.nanmax([y_true.max(), y_pred.max()])
            pad = 0.03 * (hi - lo if hi > lo else 1.0)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.8, zorder=1)

        # Stats box
        if show_stats:
            resid = y_true - y_pred
            rmse = np.sqrt(np.mean(resid**2))
            mae = np.mean(np.abs(resid))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            stats_text = "\n".join([
                stats_fmt.format(name="R²", value=r2),
                stats_fmt.format(name="MAE", value=mae),
                stats_fmt.format(name="RMSE", value=rmse),
            ])

            # Pick least-crowded corner
            xmid = np.median(y_true)
            ymid = np.median(y_pred)
            corners = {
                "upper left":  ((y_true < xmid) & (y_pred > ymid)).sum(),
                "upper right": ((y_true > xmid) & (y_pred > ymid)).sum(),
                "lower left":  ((y_true < xmid) & (y_pred < ymid)).sum(),
                "lower right": ((y_true > xmid) & (y_pred < ymid)).sum(),
            }
            loc = min(corners, key=corners.get)

            ax.text(
                0.02 if "left" in loc else 0.98,
                0.98 if "upper" in loc else 0.02,
                stats_text,
                transform=ax.transAxes,
                ha="left" if "left" in loc else "right",
                va="top" if "upper" in loc else "bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.9),
                zorder=10,
            )

        # Style
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        return ax

cal = ChemCalculatorXGRBoost("D:\\DESCARGAS\\QMugs_curatedDFT.csv")
cal.select_data(9000,"rdkit_smiles","DFT_HOMO_LUMO_GAP")
cal.smiles_to_matrix_descriptors()
cal.prepare_data()
cal.train()
cal.prediction()

cal.get_realvalue()
cal.get_predictions()
cmap_name = 'DarkBlueToDarkGold'
colors = ["#BF0615", "#EAAD12"]
puma_cmap = mcolors.LinearSegmentedColormap.from_list(
    cmap_name, colors, N=255)
cal.plot_regression(
    cal.get_realvalue(),
    cal.get_predictions(),
    color="#BF0615",
    title="Gap-parity plot (XGBoost/mordred/7200)"
)
plt.savefig("predictionXGB-MordredII.svg",format='svg')
plt.show()

sl = ["NC(=O)c1sc(-c2ccccc2)cc1N"]
cal.input_prediction(sl)

