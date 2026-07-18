import torch
import joblib
import os

MASS_KAON = 0.493677
MASS_PION = 0.139570
MASS_D0 = 1.86484

class PhysicsLossHelper:
    def __init__(self, scaler_path='scaler.pkl', device='cpu'):
        self.device = device
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"{scaler_path} not found. Ensure scaler is fitted first.")
        
        scaler = joblib.load(scaler_path)
        self.quantiles = torch.tensor(scaler.quantiles_, dtype=torch.float32, device=device) # (n_quantiles, 6)
        self.references = torch.tensor(scaler.references_, dtype=torch.float32, device=device) # (n_quantiles,)
        
    def diff_inverse_transform(self, z):
        # z: (batch, 6)
        # Clamp z to avoid extrapolating outside the learned quantiles which causes inf/nan
        z = torch.clamp(z, min=self.references[0], max=self.references[-1])
        
        idx = torch.searchsorted(self.references, z)
        idx = torch.clamp(idx, 1, len(self.references) - 1)
        
        x0 = self.references[idx - 1]
        x1 = self.references[idx]
        
        y0 = torch.gather(self.quantiles, 0, idx - 1)
        y1 = torch.gather(self.quantiles, 0, idx)
        
        # linear interpolation
        val = y0 + (y1 - y0) / (x1 - x0) * (z - x0)
        return val

    def compute_invariant_mass(self, phys):
        # phys: (batch, 6) -> [pt_K, y_K, phi_K, pt_pi, y_pi, phi_pi]
        pt1 = torch.abs(phys[:, 0]) # ensure positive pt for stability
        y1 = phys[:, 1]
        phi1 = phys[:, 2]
        m1 = MASS_KAON
        
        pt2 = torch.abs(phys[:, 3])
        y2 = phys[:, 4]
        phi2 = phys[:, 5]
        m2 = MASS_PION
        
        mT1 = torch.sqrt(pt1**2 + m1**2)
        mT2 = torch.sqrt(pt2**2 + m2**2)
        
        E1 = mT1 * torch.cosh(y1)
        E2 = mT2 * torch.cosh(y2)
        
        pz1 = mT1 * torch.sinh(y1)
        pz2 = mT2 * torch.sinh(y2)
        
        px1 = pt1 * torch.cos(phi1)
        px2 = pt2 * torch.cos(phi2)
        
        py1 = pt1 * torch.sin(phi1)
        py2 = pt2 * torch.sin(phi2)
        
        E_tot = E1 + E2
        px_tot = px1 + px2
        py_tot = py1 + py2
        pz_tot = pz1 + pz2
        
        mass_sq = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
        mass_sq = torch.clamp(mass_sq, min=1e-6) # prevent nan in sqrt
        mass = torch.sqrt(mass_sq)
        return mass

    def get_physics_loss(self, predicted_x0):
        # predicted_x0 is the estimated clean sample in standard normal space
        phys = self.diff_inverse_transform(predicted_x0)
        mass = self.compute_invariant_mass(phys)
        
        # We want the mass to be as close to MASS_D0 as possible
        loss = torch.mean((mass - MASS_D0)**2)
        return loss
