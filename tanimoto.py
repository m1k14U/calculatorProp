import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.gaussian_process.kernels import Kernel 

class FastTanimotoKernel(Kernel):
    def __init__(self):
        super().__init__()
        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        def to_bitvect(arr):
            bv = ExplicitBitVect(len(arr))
            for i,val in enumerate(arr):
                if val:
                    bv.SetBit(i)
            return bv
        X_list = [to_bitvect(fp) if isinstance(fp, np.ndarray) else fp for fp in X]
        Y_list = [to_bitvect(fp) if isinstance(fp, np.ndarray) else fp for fp in Y]

        K = np.zeros((len(X_list), len(Y_list)))
        for i, fp in enumerate(X_list):
            K[i, :] = DataStructs.BulkTanimotoSimilarity(fp, Y_list)
        if eval_gradient:
        #Comentario
            return K, np.empty((len(X_list), len(Y_list), 0))
        return K
        
    def diag(self, X):
        return np.ones(X.shape[0])
        
    def is_stationary(self):
        return False