# Este código es una alternativa al uso de GP como modelo predictivo, el uso de gpytorch es una paquetería directa de Python
# utilizada para predecir valores.

# Requisitos: torch, gpytorch, rdkit, pandas
import torch
import torch.nn as nn
import gpytorch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

# ---------- Utilidades de fingerprints ----------
def smiles_to_fp(smiles, radius=2, nBits=2048, device="cpu"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES inválido: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return torch.tensor(arr, device=device)

def build_fp_matrix(smiles_list, radius=2, nBits=2048, device="cpu"):
    fps = [smiles_to_fp(s, radius, nBits, device) for s in smiles_list]
    return torch.stack(fps)

# ---------- Kernel Tanimoto para GP ----------
class TanimotoKernel(gpytorch.kernels.Kernel):
    """
    Kernel Tanimoto para fingerprints binarios
    """
    is_stationary = False

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x1, x2, diag=False, **params):
        if x2 is None:
            x2 = x1

        x1 = x1.float()
        x2 = x2.float()

        intersection = torch.matmul(x1, x2.T)
        x1_sum = x1.sum(dim=1, keepdim=True)
        x2_sum = x2.sum(dim=1, keepdim=True).T

        union = x1_sum + x2_sum - intersection
        K = intersection / (union + self.eps)

        if diag:
            return torch.diagonal(K)

        return K

class TanimotoGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------- Modelo Exact GP ----------


# ---------- Orquestación ----------
def train_gp(model, likelihood, train_x, train_y, n_iter=100, lr=0.05):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iter {i:03d} | Loss: {loss.item():.4f}")

def predict_gp(model, likelihood, test_x):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        mean = preds.mean
        std = preds.variance.sqrt()

    return mean, std

import pandas as pd

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar datos
    df = pd.read_csv("qm9.csv")

    smiles_train = df["smiles"].iloc[:7200].tolist()
    y_train = torch.tensor(
        df["gap"].iloc[:7200].values,
        dtype=torch.float32,
        device=device
    )

    smiles_test = df["smiles"].iloc[1500:3300].tolist()

    # Fingerprints
    X_train = build_fp_matrix(smiles_train, device=device)
    X_test = build_fp_matrix(smiles_test, device=device)

    # GP
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = TanimotoGP(X_train, y_train, likelihood).to(device)

    # Train
    train_gp(model, likelihood, X_train, y_train, n_iter=150)

    # Predict
    y_pred, y_std = predict_gp(model, likelihood, X_test)

    print("Predicciones:", y_pred[:5])
    print("Incertidumbre:", y_std[:5])

# Put on a Excel
dr = pd.DataFrame({
    "gap_real": df["gap"].iloc[1500:3300].tolist(),
    "gap_pred": y_pred,
    "e_std": y_std
})
dr.to_excel("most_diverseDatPyTorch.xlsx", index=False)