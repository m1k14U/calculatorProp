# Dataset: NIH
# Prediction properties algorithm
# Model: Gaussian Process
# Kernel: Tanimoto
# Descriptor: mordred

import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns 
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tanimoto import FastTanimotoKernel
from mordred import Calculator, descriptors
from mordred.error import Error, Missing

class ChemCalculator(FastTanimotoKernel):
    def __init__(self, data_set):
        super().__init__()
        # Inicialize the classs
        self.data_set = data_set                                      # File name
        self.calculator  = Calculator(descriptors, ignore_3D = True)  # Create "Calculator" object
        self.read_set = pd.read_csv(self.data_set)                    # Load the dataset to pandas

    def show_data(self):
        #Show the dataset
        print(self.read_set)

    def select_data(self, smiles, property, number_of_data):
        # Select the range data
        self.read_set = self.read_set[0:number_of_data]               # Selecting only the data values 
        self.data_smiles = self.read_set[smiles]
        self.data_property = self.read_set[property]

        return self.data_smiles, self.data_property
    
    def smiles_to_descriptors_matrix(self):
        if self.data_smiles is not None:
            self.list_smile = [Chem.MolFromSmiles(smi) for smi in self.data_smiles]
            self.descriptor = self.calculator.pandas(self.list_smile, nproc=1)
            self.descrip_cl = self.descriptor.applymap(lambda x: np.nan if isinstance(x, (Error, Missing)) else x)
            self.descrip_cl = self.descrip_cl.dropna(axis=1, how="all")
            self.descrip_cl = self.descrip_cl.dropna(axis=1, thresh=len(self.descriptor)*0.8)
            self.features_names = self.descrip_cl.columns.tolist()
            self.X = self.descrip_cl.to_numpy()

            return self.X
        else:
            print("No data SMILES selected. Please use select_data method first.")
    
    def get_descriptors(self):
        if not self.X:
            print("Fingerprint matrix is empty. Please use matrix_fingerprints method first.")
        return self.X

    def prepare_data(self):
        # Select the percentage of train data and define the X_train, y_train, X_test and y_test
        if self.X.any() and self.data_property.any() is not None:
            self.y = self.data_property
            imputer = SimpleImputer(strategy="mean")
            self.X = imputer.fit_transform(self.X)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            return self.X_train, self.X_test, self.y_train, self.y_test
    
    def gp_train(self):
        if len(self.X_train) and len(self.y_train) == 0:
            print("Fingerprint matrix or property data is missing. Please ensure to prepare data.")
            return None

        self.kernel = FastTanimotoKernel()
        self.gp = GaussianProcessRegressor(kernel=self.kernel,alpha=1e-6, normalize_y = True) 
        self.gp.fit(self.X_train, self.y_train)
        return self.gp
    
    def gp_input_predict(self, test_smiles):
        self.list_smile = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        self.descriptor = self.calculator.pandas(self.list_smile, nproc=1)
        self.descrip_cl = self.descriptor.applymap(lambda x: np.nan if isinstance(x, (Error, Missing)) else x)
        self.descrip_cl = self.descrip_cl.dropna(axis=1, how="all")
        self.descrip_cl = self.descrip_cl.dropna(axis=1, thresh=len(self.descriptor)*0.8)
        self.descrip_cl = self.descrip_cl.reindex(columns=self.features_names)
        self.X = self.descrip_cl.to_numpy()
        self.y_pred, self.y_std = self.gp.predict(self.X, return_std=True)
        print("Predicted values:", self.y_pred)

    def gp_predict(self):
        self.y_pred, self.y_std = self.gp.predict(self.X_test, return_std=True)
        print("Predicted values:", self.y_pred)
        print("Uncertainity:",self.y_std)

    def get_realvalue(self):
        return self.y_test

    def get_predictions(self):
        return self.y_pred
    
    def get_uncertainty(self):
        return self.y_std
    
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
    
# QMugs_DFT
propCal = ChemCalculator("D:\\DESCARGAS\\QMugs_curatedDFT.csv")
propCal.select_data(smiles="rdkit_smiles", property="DFT_HOMO_LUMO_GAP", number_of_data=9000)
propCal.smiles_to_descriptors_matrix()
propCal.prepare_data()
propCal.gp_train()
propCal.gp_predict()

cmap_name = 'DarkBlueToDarkGold'
colors = ["#BF0615", "#EAAD12"]
puma_cmap = mcolors.LinearSegmentedColormap.from_list(
    cmap_name, colors, N=255)
propCal.plot_regression(
    propCal.get_realvalue(),
    propCal.get_predictions(),
    propCal.get_uncertainty(),
    cmap=puma_cmap,
    title="Gap-parity plot (GP/mordred/7200)"
)
plt.savefig("predictionGP-Mordred.svg",format='svg')
plt.show()

sl = ["NC(=O)c1sc(-c2ccccc2)cc1N"]
propCal.gp_input_predict(sl)
#COc1ccc(Cl)cc1NS(=O)(=O)c1ccc(OC)c2c1CC[C@@H](N(C)C)C2