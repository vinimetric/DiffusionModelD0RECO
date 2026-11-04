import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
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
    def __init__(self, filepath, fit_scaler=True, scaler_path='scaler.pkl',
                 mode='rest_frame', scaler_type='quantile',
                 split='train', train_ratio=0.8, seed=42):
        """
        Args:
            filepath: Path to the dataset text file.
            fit_scaler: Whether to fit and save the scaler, or load an existing one.
            scaler_path: File path to save/load the scaler.
            mode: 'rest_frame' (6D: log_pt_D0, y_D0, phi_D0, cos_theta*, phi*, M_D0)
                  or 'cartesian' (8D: px_K, py_K, pz_K, E_K, px_pi, py_pi, pz_pi, E_pi).
            scaler_type: 'quantile' (QuantileTransformer) or 'standard' (StandardScaler).
            split: 'train' (uses train_ratio), 'test' (uses 1-train_ratio), or 'all'.
            train_ratio: Fraction of events for training (default 0.8 = 80%).
            seed: Random seed for deterministic train/test split.
        """
        self.mode = mode
        self.scaler_type = scaler_type
        self.split = split

        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None)
            if df.shape[1] != 8:
                df = pd.read_csv(filepath, sep=',', header=None)
        except Exception:
            df = pd.read_csv(filepath, sep=',', header=None)

        if df.shape[1] > 8:
            df = df.iloc[:, :8]

        df.columns = ['px1', 'px2', 'py1', 'py2', 'pz1', 'pz2', 'E1', 'E2']

        # ------------------------------------------------------------------
        # Train / Test split (80/20) with deterministic seed
        # ------------------------------------------------------------------
        n_total = len(df)
        n_train = int(train_ratio * n_total)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)

        train_indices = indices[:n_train]
        test_indices  = indices[n_train:]

        # Export raw test set to disk for evaluate.py if not already present
        test_export_path = 'data_D0_test.txt'
        if not os.path.exists(test_export_path):
            df_test = df.iloc[test_indices]
            df_test.to_csv(test_export_path, sep=' ', index=False, header=False)
            print(f"[Dataset] Exported test partition ({len(df_test)} events, {(1-train_ratio)*100:.0f}%) to {test_export_path}")

        if split == 'train':
            df = df.iloc[train_indices].reset_index(drop=True)
        elif split == 'test':
            df = df.iloc[test_indices].reset_index(drop=True)
        elif split == 'all':
            pass
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'train', 'test', or 'all'.")

        print(f"[Dataset] Split: '{split}' | Events loaded: {len(df)} / {n_total} ({len(df)/n_total*100:.1f}%)")

        px1 = df['px1'].values; py1 = df['py1'].values
        pz1 = df['pz1'].values; E1  = df['E1'].values
        px2 = df['px2'].values; py2 = df['py2'].values
        pz2 = df['pz2'].values; E2  = df['E2'].values

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

        # Order daughters so track1 is always Kaon and track2 is always Pion
        px_K = np.where(is_trk1_kaon, px1, px2)
        py_K = np.where(is_trk1_kaon, py1, py2)
        pz_K = np.where(is_trk1_kaon, pz1, pz2)
        E_K  = np.where(is_trk1_kaon, E1,  E2)

        px_pi = np.where(is_trk1_kaon, px2, px1)
        py_pi = np.where(is_trk1_kaon, py2, py1)
        pz_pi = np.where(is_trk1_kaon, pz2, pz1)
        E_pi  = np.where(is_trk1_kaon, E2,  E1)

        if self.mode == 'cartesian':
            # 8D Cartesian representation: [px_K, py_K, pz_K, E_K, px_pi, py_pi, pz_pi, E_pi]
            data = np.column_stack([
                px_K, py_K, pz_K, E_K,
                px_pi, py_pi, pz_pi, E_pi
            ])
            self.feature_names = ['px_K', 'py_K', 'pz_K', 'E_K', 'px_pi', 'py_pi', 'pz_pi', 'E_pi']
        elif self.mode == 'rest_frame':
            # ------------------------------------------------------------------
            # Compute D0 4-momentum
            # ------------------------------------------------------------------
            px_D0 = px_K + px_pi
            py_D0 = py_K + py_pi
            pz_D0 = pz_K + pz_pi
            E_D0  = E_K  + E_pi

            pt_D0  = np.sqrt(px_D0**2 + py_D0**2)
            M_D0   = np.sqrt(np.maximum(E_D0**2 - px_D0**2 - py_D0**2 - pz_D0**2, 1e-10))
            phi_D0 = np.arctan2(py_D0, px_D0)

            eps = 1e-10
            y_D0_num = E_D0 + pz_D0
            y_D0_den = np.where(E_D0 - pz_D0 < eps, eps, E_D0 - pz_D0)
            y_D0 = 0.5 * np.log(y_D0_num / y_D0_den)

            log_pt_D0 = np.log(np.maximum(pt_D0, eps))

            # Boost the kaon to the D0 rest frame to get decay angles
            px_K_star, py_K_star, pz_K_star, _ = boost_to_rest_frame(
                px_K, py_K, pz_K, E_K, px_D0, py_D0, pz_D0, E_D0
            )

            pK_star_mag = np.sqrt(px_K_star**2 + py_K_star**2 + pz_K_star**2)
            pK_star_mag = np.maximum(pK_star_mag, eps)

            cos_theta_star = pz_K_star / pK_star_mag
            phi_star       = np.arctan2(py_K_star, px_K_star)

            # 6D Rest-Frame representation
            data = np.column_stack([
                log_pt_D0,
                y_D0,
                phi_D0,
                cos_theta_star,
                phi_star,
                M_D0,
            ])
            self.feature_names = ['log_pt_D0', 'y_D0', 'phi_D0', 'cos_theta_K*', 'phi_K*', 'M_D0']
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'rest_frame' or 'cartesian'.")

        self.data = data
        self.feature_dim = self.data.shape[1]

        # ------------------------------------------------------------------
        # Scaler selection & fitting (fits only on train to prevent leakage)
        # ------------------------------------------------------------------
        if fit_scaler:
            if self.scaler_type == 'quantile':
                self.scaler = QuantileTransformer(
                    output_distribution='normal', n_quantiles=10000, random_state=42
                )
            elif self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {self.scaler_type}. Choose 'quantile' or 'standard'.")

            self.data = self.scaler.fit_transform(self.data)
            
            # Save scaler and metadata
            scaler_meta = {
                'scaler': self.scaler,
                'mode': self.mode,
                'scaler_type': self.scaler_type,
                'feature_dim': self.feature_dim,
                'feature_names': self.feature_names,
                'train_ratio': train_ratio,
                'seed': seed
            }
            joblib.dump(scaler_meta, scaler_path)
            print(f"[Dataset] Scaler fit on {split} ({len(self.data)} events) and saved to {scaler_path}")
            print(f"[Dataset] Mode: {self.mode} | Scaler: {self.scaler_type} | Dim: {self.feature_dim}")
        else:
            if os.path.exists(scaler_path):
                scaler_meta = joblib.load(scaler_path)
                if isinstance(scaler_meta, dict) and 'scaler' in scaler_meta:
                    self.scaler = scaler_meta['scaler']
                else:
                    self.scaler = scaler_meta
                self.data = self.scaler.transform(self.data)
            else:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(filepath, batch_size=256, shuffle=True, fit_scaler=True,
                   scaler_path='scaler.pkl', mode='rest_frame', scaler_type='quantile',
                   split='train', train_ratio=0.8, seed=42):
    dataset = D0Dataset(
        filepath, fit_scaler=fit_scaler, scaler_path=scaler_path,
        mode=mode, scaler_type=scaler_type,
        split=split, train_ratio=train_ratio, seed=seed
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
