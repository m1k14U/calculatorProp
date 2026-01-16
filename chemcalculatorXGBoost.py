import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import descriptastorus
from descriptastorus.descriptors import  MakeGenerator

class ChemCalculatorXGRBoost:
    def __init__(self, csv_file):
        self.valid_smiles = []
        self.X = []
        self.generator = MakeGenerator(("RDKit2DNormalized","Morgan3Counts"))
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
        print(self.data)

    def select_data(self,number_of_data,feature:str,property:str):
        self.data = self.data[1:number_of_data]
        self.smiles_data = self.data[feature]
        self.property_data = self.data[property]

    def smiles_to_matrix_descriptors(self):
        if self.smiles_data is not None:
            for smi in self.smiles_data:
                self.descrip = self.generator.process(smi)
                if self.descrip:
                    self.vecto = np.array(self.descrip[1::])
                    self.X.append(self.vecto)
                    self.valid_smiles.append(smi)
            self.X = np.array(self.X)
            return self.X
        else:
            print("Could not be possible to create a matrix. Make sure you select the molecules features with select_data()")

    def input_smiles_to_matrix(self,lisml):
        self.X_new = []
        for smi in lisml:
            self.descrip = self.generator.process(smi)
            if self.descrip:
                self.vecto = np.array(self.descrip[1::])
                self.X_new.append(self.vecto)
                self.valid_smiles.append(smi)
        self.X_new = np.array(self.X_new)
        return self.X_new

    def get_shape_matrix_descriptors(self):
        print(self.X.shape)

    def prepare_data(self):
        if self.X and self.property_data:
            self.y = self.property_data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            return self.X_train, self.X_test, self.y_train, self.y_test
        
    def train(self):
        if self.X_train and self.y_train is not None:
            self.model.fit(self.X_train,self.y_train)

            return self.model
    
    def prediction(self):
        if self.X_test and self.y_test is not None:
            self.y_pred = self.model.predict(self.X_test)
            print("MAE: ",mean_absolute_error(self.y_test, self.y_pred))
            print("R2: ", r2_score(self.y_test, self.y_pred))
            
            return self.model
        else:
            print("No data for test!")
    
    def input_prediction(self,smileList):
        self.input_smiles_to_matrix(smileList)
        self.y_new = self.model.predict(self.X_new)

        print(self.y_new)

    def plot_regression(self, y_true, y_pred, y_std=None, ideal_line=True, cmap="viridis", s=30, alpha=0.9, ax=None, xlabel="Actual", 
                        ylabel="Predicted", title=None, equal_aspect=True, show_stats=True, stats_fmt="{name} = {value:.3f}", add_colorbar=True,):
        
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape after ravel().")

        if y_std is not None:
            y_std = np.asarray(y_std).ravel()
            if y_std.shape != y_true.shape:
                raise ValueError("y_std must have the same shape as y_true/y_pred.")

        # Drop non-finite rows
        if y_std is None:
            m = np.isfinite(y_true) & np.isfinite(y_pred)
        else:
            m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)

        y_true = y_true[m]
        y_pred = y_pred[m]
        if y_std is not None:
            y_std = y_std[m]

        # Create axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
        else:
            fig = ax.figure

        # Scatter
        if y_std is None:
            sc = ax.scatter(y_true, y_pred, s=s, alpha=alpha, edgecolor="none", zorder=3)
        else:
            sc = ax.scatter(
                y_true, y_pred,
                c=y_std,
                cmap=cmap,
                s=s,
                alpha=alpha,
                edgecolor="none",
                zorder=3,
            )

        # Reference line y=x over data range
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
                    # Pick least-crowded corner using medians in parity space
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

        # Colorbar
        if add_colorbar and (y_std is not None):
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("Predictive uncertainty (std)")

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

cal = ChemCalculatorXGRBoost("qm9.csv")
cal.select_data(9000,"smiles","gap")
cal.smiles_to_matrix_descriptors()
cal.prepare_data()
cal.train()
cal.prediction()
"""
## Puma cmap 
cmap_name = 'DarkBlueToDarkGold'
colors = ['#00008B', '#B8860B']

puma_cmap = mcolors.LinearSegmentedColormap.from_list(
    cmap_name, colors, N=255)

cal.plot_regression(
    list(data_test["gap"][15001:20000]),
    cal.get_predictions(),
    cal.get_uncertainty(),
    cmap=puma_cmap,
    title="Gap-parity plot (colored by uncertainty)"
)
plt.savefig("prediction.svg",format='svg')
plt.show()
"""