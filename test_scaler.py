import torch
import joblib
import numpy as np

def test_diff_scaler():
    scaler = joblib.load('scaler.pkl')
    quantiles = torch.tensor(scaler.quantiles_, dtype=torch.float32) # (n_quantiles, 6)
    references = torch.tensor(scaler.references_, dtype=torch.float32) # (n_quantiles,)

    def diff_inverse_transform(z):
        idx = torch.searchsorted(references, z)
        idx = torch.clamp(idx, 1, len(references) - 1)
        
        x0 = references[idx - 1]
        x1 = references[idx]
        
        y0 = torch.gather(quantiles, 0, idx - 1)
        y1 = torch.gather(quantiles, 0, idx)
        
        val = y0 + (y1 - y0) / (x1 - x0) * (z - x0)
        return val

    z = torch.randn(2, 6, requires_grad=True)
    phys = diff_inverse_transform(z)
    mass = phys.sum()
    mass.backward()
    print(z.grad)
    print('Differentiable scaler works!')

if __name__ == "__main__":
    test_diff_scaler()
