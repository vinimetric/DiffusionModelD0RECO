import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import time
import os
import matplotlib.pyplot as plt
from model import DDPM_MLP
from diffusion import Diffusion
from dataset import MASS_KAON, MASS_PION, MASS_D0_NOM


def boost_from_rest_frame(px_star, py_star, pz_star, E_star, px_D0, py_D0, pz_D0, E_D0):
    """Boost a 4-vector from the D0 rest frame back to the lab frame."""
    eps = 1e-10
    p_D0_mag = np.sqrt(px_D0**2 + py_D0**2 + pz_D0**2)
    M_D0     = np.sqrt(np.maximum(E_D0**2 - p_D0_mag**2, eps))
    p_D0_mag = np.maximum(p_D0_mag, eps)

    nx = px_D0 / p_D0_mag
    ny = py_D0 / p_D0_mag
    nz = pz_D0 / p_D0_mag

    beta  = p_D0_mag / E_D0
    gamma = E_D0 / M_D0

    # Longitudinal component in rest frame
    p_L_star = px_star * nx + py_star * ny + pz_star * nz

    # Transverse components (unchanged)
    px_T = px_star - p_L_star * nx
    py_T = py_star - p_L_star * ny
    pz_T = pz_star - p_L_star * nz

    # Boost back (opposite direction: +beta)
    p_L_lab = gamma * (p_L_star + beta * E_star)
    E_lab   = gamma * (E_star   + beta * p_L_star)

    px_lab = px_T + p_L_lab * nx
    py_lab = py_T + p_L_lab * ny
    pz_lab = pz_T + p_L_lab * nz

    return px_lab, py_lab, pz_lab, E_lab


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine paths and feature dimension
    feature_dim = 8 if args.mode == 'cartesian' else 6
    scaler_path = args.scaler_path if args.scaler_path else f"scaler_{args.mode}_{args.scaler_type}.pkl"
    if not os.path.exists(scaler_path) and os.path.exists('scaler.pkl') and args.mode == 'rest_frame' and args.scaler_type == 'quantile':
        scaler_path = 'scaler.pkl'

    model_path = args.model_path if args.model_path else f"checkpoints/ddpm_{args.mode}_{args.scaler_type}_final.pt"
    if not os.path.exists(model_path) and os.path.exists('checkpoints/ddpm_final.pt') and args.mode == 'rest_frame':
        model_path = 'checkpoints/ddpm_final.pt'

    output_path = args.output_path if args.output_path else f"generated_samples_{args.mode}_{args.scaler_type}.txt"

    print(f"============================================================")
    print(f" Generation Configuration:")
    print(f"   Mode:              {args.mode} (Dimension: {feature_dim})")
    print(f"   Scaler Type:       {args.scaler_type}")
    print(f"   Model Weights:     {model_path}")
    print(f"   Scaler Path:       {scaler_path}")
    print(f"   Output Path:       {output_path}")
    print(f"   Samples:           {args.num_samples} | Sampler: {args.sampler}")
    print(f"============================================================")

    model = DDPM_MLP(feature_dim=feature_dim, hidden_dim=256, time_emb_dim=256, num_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    diffusion = Diffusion(timesteps=args.timesteps, device=device, schedule='linear')

    if args.benchmark:
        print("Running generation speed benchmark...")
        batch_sizes = [1, 10, 100, 1000, 50000]
        times_per_event = []
        for bs in batch_sizes:
            start_time = time.time()
            _ = diffusion.sample(model, n_samples=bs, feature_dim=feature_dim)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            tpe = (end_time - start_time) / bs
            times_per_event.append(tpe)
            print(f"Batch Size: {bs:5d} | Total Time: {end_time-start_time:.4f}s | Time/Event: {tpe:.6f}s")

        plt.figure(figsize=(8, 6))
        plt.plot(batch_sizes, times_per_event, marker='o')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Batch Size'); plt.ylabel('Time per Event (s)')
        plt.title('DDPM Generation Speed vs Batch Size')
        plt.grid(True, which='both', ls='--')
        plt.savefig(f'generation_speed_{args.mode}_{args.scaler_type}.png')
        plt.close()

    print(f"Generating {args.num_samples} samples...")
    start_time = time.time()

    if args.sampler == 'ddim':
        generated_scaled = diffusion.sample_ddim(
            model, n_samples=args.num_samples, feature_dim=feature_dim,
            ddim_steps=args.ddim_steps, eta=args.eta
        )
    else:
        generated_scaled = diffusion.sample(model, n_samples=args.num_samples, feature_dim=feature_dim)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Generation took {end_time-start_time:.2f}s ({(end_time-start_time)/args.num_samples:.6f} s/event)")

    generated_scaled = generated_scaled.cpu().numpy()

    # Load scaler
    scaler_obj = joblib.load(scaler_path)
    scaler = scaler_obj['scaler'] if isinstance(scaler_obj, dict) and 'scaler' in scaler_obj else scaler_obj

    eps = 1e-10

    if args.mode == 'cartesian':
        # ------------------------------------------------------------------ #
        # Case A: Cartesian Mode (8D -> px, py, pz, E for both daughters)
        # ------------------------------------------------------------------ #
        gen = scaler.inverse_transform(generated_scaled)
        px_K  = gen[:, 0]
        py_K  = gen[:, 1]
        pz_K  = gen[:, 2]
        E_K   = gen[:, 3]

        px_pi = gen[:, 4]
        py_pi = gen[:, 5]
        pz_pi = gen[:, 6]
        E_pi  = gen[:, 7]

        def to_kinematics(px, py, pz, E):
            pt = np.sqrt(px**2 + py**2)
            phi = np.arctan2(py, px)
            p_mag_sq = px**2 + py**2 + pz**2
            m = np.sqrt(np.maximum(E**2 - p_mag_sq, 0.0))
            num = np.where(E + pz < eps, eps, E + pz)
            den = np.where(E - pz < eps, eps, E - pz)
            y = 0.5 * np.log(num / den)
            return pt, y, phi, m

        pt_K,  y_K,  phi_K,  m_K  = to_kinematics(px_K,  py_K,  pz_K,  E_K)
        pt_pi, y_pi, phi_pi, m_pi = to_kinematics(px_pi, py_pi, pz_pi, E_pi)

        # Invariant mass of D0
        E_tot  = E_K  + E_pi
        px_tot = px_K + px_pi
        py_tot = py_K + py_pi
        pz_tot = pz_K + pz_pi
        M_D0   = np.sqrt(np.maximum(E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2, 0.0))

        df = pd.DataFrame({
            'pt1': pt_K,  'y1': y_K,  'phi1': phi_K,  'm1': m_K,
            'pt2': pt_pi, 'y2': y_pi, 'phi2': phi_pi, 'm2': m_pi,
        })[['pt1', 'y1', 'phi1', 'm1', 'pt2', 'y2', 'phi2', 'm2']]

    else:
        # ------------------------------------------------------------------ #
        # Case B: Rest-Frame Mode (6D -> Physics-Informed Reconstruction)
        # ------------------------------------------------------------------ #
        if args.scaler_type == 'quantile' and args.smooth_ydD0_sigma > 0:
            jitter = np.random.normal(0, args.smooth_ydD0_sigma, len(generated_scaled))
            generated_scaled[:, 1] += jitter  # index 1 = y_D0

        gen = scaler.inverse_transform(generated_scaled)

        log_pt_D0      = gen[:, 0]
        y_D0           = gen[:, 1]
        phi_D0         = gen[:, 2]
        cos_theta_star = np.clip(gen[:, 3], -1.0, 1.0)
        phi_star       = gen[:, 4]
        M_D0           = np.abs(gen[:, 5])

        # D0 lab 4-momentum
        pt_D0  = np.exp(log_pt_D0)
        mT_D0  = np.sqrt(pt_D0**2 + M_D0**2)
        px_D0  = pt_D0 * np.cos(phi_D0)
        py_D0  = pt_D0 * np.sin(phi_D0)
        pz_D0  = mT_D0 * np.sinh(y_D0)
        E_D0   = mT_D0 * np.cosh(y_D0)

        # Källén function for rest-frame momentum magnitude
        mK2, mpi2 = MASS_KAON**2, MASS_PION**2
        M2 = M_D0**2
        kallen = M2**2 + mK2**2 + mpi2**2 - 2*M2*mK2 - 2*M2*mpi2 - 2*mK2*mpi2
        kallen = np.maximum(kallen, 0.0)
        p_star_mag = np.sqrt(kallen) / (2.0 * M_D0)

        # Kaon rest-frame components
        sin_theta_star = np.sqrt(np.maximum(1.0 - cos_theta_star**2, 0.0))
        px_K_star = p_star_mag * sin_theta_star * np.cos(phi_star)
        py_K_star = p_star_mag * sin_theta_star * np.sin(phi_star)
        pz_K_star = p_star_mag * cos_theta_star
        E_K_star  = np.sqrt(mK2 + p_star_mag**2)

        # Boost Kaon to lab frame
        px_K, py_K, pz_K, E_K = boost_from_rest_frame(
            px_K_star, py_K_star, pz_K_star, E_K_star,
            px_D0, py_D0, pz_D0, E_D0
        )

        # Pion via exact 4-momentum conservation
        px_pi = px_D0 - px_K
        py_pi = py_D0 - py_K
        pz_pi = pz_D0 - pz_K
        E_pi  = E_D0  - E_K

        def to_pt_y_phi(px, py, pz, E):
            pt  = np.sqrt(px**2 + py**2)
            phi = np.arctan2(py, px)
            num = np.where(E + pz < eps, eps, E + pz)
            den = np.where(E - pz < eps, eps, E - pz)
            y   = 0.5 * np.log(num / den)
            return pt, y, phi

        pt_K,  y_K,  phi_K  = to_pt_y_phi(px_K,  py_K,  pz_K,  E_K)
        pt_pi, y_pi, phi_pi = to_pt_y_phi(px_pi, py_pi, pz_pi, E_pi)

        df = pd.DataFrame({
            'pt1': pt_K,  'y1': y_K,  'phi1': phi_K,  'm1': MASS_KAON,
            'pt2': pt_pi, 'y2': y_pi, 'phi2': phi_pi, 'm2': MASS_PION,
        })[['pt1', 'y1', 'phi1', 'm1', 'pt2', 'y2', 'phi2', 'm2']]

    # Save to disk
    df.to_csv(output_path, sep=' ', index=False, header=False)
    # Also overwrite generated_samples.txt for backward compatibility
    df.to_csv('generated_samples.txt', sep=' ', index=False, header=False)

    print(f"\nGenerated samples saved to: {output_path}")
    print(f"M_D0 generated: mean = {M_D0.mean():.4f} GeV, std = {M_D0.std()*1000:.2f} MeV")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate D0 samples using trained DDPM")
    parser.add_argument('--mode',        type=str, default='rest_frame', choices=['rest_frame', 'cartesian'],
                        help="Model representation: 'rest_frame' or 'cartesian'")
    parser.add_argument('--scaler_type', type=str, default='quantile', choices=['quantile', 'standard'],
                        help="Scaler: 'quantile' or 'standard'")
    parser.add_argument('--model_path',  type=str, default=None,
                        help="Path to trained model weights (.pt)")
    parser.add_argument('--scaler_path', type=str, default=None,
                        help="Path to scaler (.pkl)")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path to save generated samples")
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--timesteps',   type=int, default=1000)
    parser.add_argument('--benchmark',   action='store_true')
    parser.add_argument('--sampler',     type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Sampler: ddim (deterministic) or ddpm (stochastic)')
    parser.add_argument('--ddim_steps',  type=int, default=200,
                        help='Number of DDIM denoising steps')
    parser.add_argument('--eta',         type=float, default=0.0,
                        help='DDIM eta (0=deterministic)')
    parser.add_argument('--smooth_ydD0_sigma', type=float, default=0.03,
                        help='Gaussian smoothing on y_D0 for quantile transformer artifact')
    args = parser.parse_args()

    generate(args)
