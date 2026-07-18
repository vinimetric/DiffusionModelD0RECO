import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import joblib
import os

# D0 -> K + Pi: fixed daughter masses (GeV)
MASS_KAON = 0.493677
MASS_PION = 0.139570
MASS_D0_NOM = 1.86484  # Nominal D0 mass (GeV)

# =============================================================================
#  Lorentz boost utilities
# =============================================================================
def boost_to_rest_frame(px, py, pz, E, px_D0, py_D0, pz_D0, E_D0):
    """Boost a 4-vector (px, py, pz, E) to the D0 rest frame (arrays)."""
    eps = 1e-10
    p_D0_mag = np.sqrt(px_D0**2 + py_D0**2 + pz_D0**2)
    M_D0 = np.sqrt(np.maximum(E_D0**2 - p_D0_mag**2, eps))
    p_D0_mag = np.maximum(p_D0_mag, eps)

    # Unit vector along D0 direction
    nx = px_D0 / p_D0_mag
    ny = py_D0 / p_D0_mag
    nz = pz_D0 / p_D0_mag

    beta  = p_D0_mag / E_D0          # |v| of D0 in lab
    gamma = E_D0 / M_D0              # Lorentz factor

    # Longitudinal momentum of daughter along D0 direction
    p_L   = px * nx + py * ny + pz * nz

    # Transverse momentum (unchanged by the boost)
    px_T  = px - p_L * nx
    py_T  = py - p_L * ny
    pz_T  = pz - p_L * nz

    # Boost the longitudinal momentum and energy
    p_L_star = gamma * (p_L - beta * E)
    E_star   = gamma * (E   - beta * p_L)

    px_star = px_T + p_L_star * nx
    py_star = py_T + p_L_star * ny
    pz_star = pz_T + p_L_star * nz

    return px_star, py_star, pz_star, E_star


class D0Dataset(Dataset):
    def __init__(self, filepath, fit_scaler=True, scaler_path='scaler.pkl'):
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None)
            if df.shape[1] != 8:
                df = pd.read_csv(filepath, sep=',', header=None)
        except Exception:
            df = pd.read_csv(filepath, sep=',', header=None)

        if df.shape[1] > 8:
            df = df.iloc[:, :8]

        df.columns = ['px1', 'px2', 'py1', 'py2', 'pz1', 'pz2', 'E1', 'E2']

        px1 = df['px1'].values; py1 = df['py1'].values
        pz1 = df['pz1'].values; E1  = df['E1'].values
        px2 = df['px2'].values; py2 = df['py2'].values
        pz2 = df['pz2'].values; E2  = df['E2'].values

        # ------------------------------------------------------------------
        # Compute D0 4-momentum
        # ------------------------------------------------------------------
        px_D0 = px1 + px2
        py_D0 = py1 + py2
        pz_D0 = pz1 + pz2
        E_D0  = E1  + E2

        pt_D0  = np.sqrt(px_D0**2 + py_D0**2)
        M_D0   = np.sqrt(np.maximum(E_D0**2 - px_D0**2 - py_D0**2 - pz_D0**2, 1e-10))
        phi_D0 = np.arctan2(py_D0, px_D0)

        eps = 1e-10
        y_D0_num = E_D0 + pz_D0
        y_D0_den = np.where(E_D0 - pz_D0 < eps, eps, E_D0 - pz_D0)
        y_D0 = 0.5 * np.log(y_D0_num / y_D0_den)

        log_pt_D0 = np.log(np.maximum(pt_D0, eps))

        # ------------------------------------------------------------------
        # Identify kaon per event by invariant mass (data is 50/50 mixed)
        # The kaon is always the higher-mass track (m_K=0.494 vs m_pi=0.140)
        # ------------------------------------------------------------------
        p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)
        p2_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
        m1_sq  = np.maximum(E1**2 - p1_mag**2, 0)
        m2_sq  = np.maximum(E2**2 - p2_mag**2, 0)

        # track1 is kaon when m1 > m2
        is_trk1_kaon = m1_sq > m2_sq

        # Select kaon components per event
        px_K = np.where(is_trk1_kaon, px1, px2)
        py_K = np.where(is_trk1_kaon, py1, py2)
        pz_K = np.where(is_trk1_kaon, pz1, pz2)
        E_K  = np.where(is_trk1_kaon, E1,  E2)

        # ------------------------------------------------------------------
        # Boost the kaon to the D0 rest frame to get decay angles
        # ------------------------------------------------------------------
        px_K_star, py_K_star, pz_K_star, _ = boost_to_rest_frame(
            px_K, py_K, pz_K, E_K, px_D0, py_D0, pz_D0, E_D0
        )

        pK_star_mag = np.sqrt(px_K_star**2 + py_K_star**2 + pz_K_star**2)
        pK_star_mag = np.maximum(pK_star_mag, eps)

        cos_theta_star = pz_K_star / pK_star_mag
        phi_star       = np.arctan2(py_K_star, px_K_star)

        # ------------------------------------------------------------------
        # Pack features: [log_pt_D0, y_D0, phi_D0, cos_theta*, phi*, M_D0]
        # ------------------------------------------------------------------
        data = np.column_stack([
            log_pt_D0,
            y_D0,
            phi_D0,
            cos_theta_star,
            phi_star,
            M_D0,
        ])

        self.data = data

        if fit_scaler:
            self.scaler = QuantileTransformer(
                output_distribution='normal', n_quantiles=10000, random_state=42
            )
            self.data = self.scaler.fit_transform(self.data)
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
            print(f"Feature order: [log_pt_D0, y_D0, phi_D0, cos_theta_K*, phi_K*, M_D0]")
            print(f"Fixed daughter masses: m_K = {MASS_KAON:.6f} GeV, m_pi = {MASS_PION:.6f} GeV")
            print(f"M_D0 range: [{M_D0.min():.4f}, {M_D0.max():.4f}] GeV, mean = {M_D0.mean():.4f} GeV")
        else:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.data = self.scaler.transform(self.data)
            else:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(filepath, batch_size=256, shuffle=True, fit_scaler=True, scaler_path='scaler.pkl'):
    dataset = D0Dataset(filepath, fit_scaler=fit_scaler, scaler_path=scaler_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
