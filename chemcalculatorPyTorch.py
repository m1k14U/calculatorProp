# Requisitos: torch, gpytorch, rdkit, pandas
import torch
import gpytorch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------- Utilidades de fingerprints ----------
def smiles_to_bit_tensor(smiles: str, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    # Convertir a tensor float (0/1)
    arr = torch.tensor([int(b) for b in fp.ToBitString()], dtype=torch.float32)
    return arr

def build_fingerprint_matrix(smiles_list, radius=2, nBits=2048, device="cpu"):
    fps = [smiles_to_bit_tensor(s, radius=radius, nBits=nBits) for s in smiles_list]
    X = torch.stack(fps).to(device)  # [N, nBits] binario 0/1
    return X

# ---------- Kernel Tanimoto para GP ----------
class TanimotoKernel(gpytorch.kernels.Kernel):
    """
    Kernel basado en similitud de Tanimoto para vectores binarios (0/1).
    K(x,y) = (x·y) / (sum(x) + sum(y) - x·y)
    Opcionalmente con escala de salida (variance) y temperatura para suavizar.
    """
    has_lengthscale = False

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            name="raw_temperature", parameter=torch.nn.Parameter(torch.zeros(1))
        )
        # Transformaciones positivas
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())
        self.register_constraint("raw_temperature", gpytorch.constraints.Positive())
        self.eps = eps

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @property
    def temperature(self):
        # Temperatura ~1 suaviza el denominador; puedes fijarla o aprenderla
        return self.raw_temperature_constraint.transform(self.raw_temperature) + 1.0

    def forward(self, x, y, **params):
        # x: [N, D], y: [M, D], binario 0/1 (float)
        dot = x @ y.transpose(-1, -2)                       # [N, M]
        sum_x = x.sum(dim=-1, keepdim=True)                 # [N, 1]
        sum_y = y.sum(dim=-1, keepdim=True).transpose(-1, -2)  # [1, M] -> [M] transpuesto
        denom = (sum_x + sum_y - dot) * self.temperature
        tanimoto = dot / denom.clamp(min=self.eps)          # [N, M]
        return self.variance * tanimoto

# ---------- Modelo Exact GP ----------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---------- Orquestación ----------
class ChemCalculatorGP:
    def __init__(self, csv_path, device="cpu"):
        self.df = pd.read_csv(csv_path)
        self.device = device
        self.train_x = None
        self.train_y = None
        self.model = None
        self.likelihood = None
        self.kernel = None

    def select_data(self, smiles_col, property_col, number_of_data):
        sel = self.df.iloc[:number_of_data]
        self.smiles = sel[smiles_col].tolist()
        self.targets = torch.tensor(sel[property_col].values, dtype=torch.float32, device=self.device)
        return self.smiles, self.targets

    def build_train(self, radius=2, nBits=2048):
        self.train_x = build_fingerprint_matrix(self.smiles, radius=radius, nBits=nBits, device=self.device)
        self.train_y = self.targets

    def init_model(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.kernel = TanimotoKernel().to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, self.kernel).to(self.device)

    def train(self, num_iter=200, lr=0.05):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
        ], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(1, num_iter + 1):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if i % 20 == 0:
                print(f"Iter {i}/{num_iter} - Loss: {loss.item():.4f} | "
                      f"variance: {self.model.covar_module.variance.item():.4f}")
            optimizer.step()

    def predict(self, smiles_list, radius=2, nBits=2048, batch_size=4096):
        self.model.eval()
        self.likelihood.eval()
        X_test = build_fingerprint_matrix(smiles_list, radius=radius, nBits=nBits, device=self.device)

        preds_mean = []
        preds_std = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Para tamaños grandes, hacer batches
            for i in range(0, X_test.size(0), batch_size):
                xb = X_test[i:i+batch_size]
                # Predicción tipo exact GP: usar train kernel para cross-covariances
                # gpytorch maneja internamente K(xb, xb) y K(train, xb)
                pred = self.likelihood(self.model(xb))
                mean = pred.mean
                var = pred.variance.clamp_min(1e-12)
                preds_mean.append(mean.cpu())
                preds_std.append(var.sqrt().cpu())

        mean_all = torch.cat(preds_mean).numpy()
        std_all = torch.cat(preds_std).numpy()
        return mean_all, std_all

# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpcal = ChemCalculatorGP("qm9.csv", device=device)
    gpcal.select_data(smiles_col="smiles", property_col="gap", number_of_data=15000)
    gpcal.build_train(radius=2, nBits=2048)
    gpcal.init_model()
    gpcal.train(num_iter=300, lr=0.05)

    # Test
    data_test = pd.read_csv("qm9.csv")
    smiles_test = data_test["smiles"].iloc[15001:30000].tolist()
    y_pred, y_std = gpcal.predict(smiles_test, radius=2, nBits=2048, batch_size=2048)
    print("Predicted values:", y_pred)
    print("Uncertainty:", y_std)
