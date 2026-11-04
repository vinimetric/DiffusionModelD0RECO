import os
import pandas as pd
import numpy as np
import argparse
import jetnet
import matplotlib
matplotlib.use('Agg') #  algum bug maluco aqui entre Matplotlib and Windows multiprocessing. safe import

# deixando bonito os plots
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use("default")
plt.style.use(hep.style.CMS)
plt.rcParams['font.size'] = 12

def evaluate(args):
    # Load real data
    try:
        real_df = pd.read_csv(args.real_data, sep=r'\s+', header=None)
        if real_df.shape[1] != 8:
            real_df = pd.read_csv(args.real_data, sep=',', header=None)
    except Exception as e:
        real_df = pd.read_csv(args.real_data, sep=',', header=None)
    
    if real_df.shape[1] > 8:
        real_df = real_df.iloc[:, :8]
        
    real_df.columns = ['px1', 'px2', 'py1', 'py2', 'pz1', 'pz2', 'E1', 'E2']
    print(f"[Evaluate] Comparing against real dataset: '{args.real_data}' ({len(real_df)} events)")
    
    # Convert real_df to pt, y, phi, m
    real_converted = {}
    for i, p_type in enumerate(['1', '2']):
        px, py, pz, E = real_df[f'px{p_type}'].values, real_df[f'py{p_type}'].values, real_df[f'pz{p_type}'].values, real_df[f'E{p_type}'].values
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        y = 0.5 * np.log((E + pz + 1e-8) / (E - pz + 1e-8))
        phi = np.arctan2(py, px)
        m = np.sqrt(np.maximum(E**2 - p**2, 0))
        real_converted[f'pt{p_type}'] = pt
        real_converted[f'y{p_type}'] = y
        real_converted[f'phi{p_type}'] = phi
        real_converted[f'm{p_type}'] = m
        real_converted[f'px{p_type}'] = px
        real_converted[f'py{p_type}'] = py
        real_converted[f'pz{p_type}'] = pz
        real_converted[f'E{p_type}'] = E
    real_df = pd.DataFrame(real_converted)

    # ------------------------------------------------------------------
    # Re-order real data so track1=kaon, track2=pion (matching generated)
    # Real data has kaon/pion randomly assigned per event (50/50 mixed)
    # ------------------------------------------------------------------
    is_trk1_kaon = real_df['m1'] > real_df['m2']
    for col in ['pt', 'y', 'phi', 'm', 'px', 'py', 'pz', 'E']:
        c1, c2 = f'{col}1', f'{col}2'
        orig1 = real_df[c1].values.copy()
        orig2 = real_df[c2].values.copy()
        # Where track1 is pion (not kaon): swap
        real_df[c1] = np.where(is_trk1_kaon, orig1, orig2)
        real_df[c2] = np.where(is_trk1_kaon, orig2, orig1)
    print(f"Real data: {is_trk1_kaon.sum()} events had kaon as track1, "
          f"{(~is_trk1_kaon).sum()} were swapped. Now track1=kaon, track2=pion.")
    
    # Load fake data
    fake_df = pd.read_csv(args.fake_data, sep=r'\s+', header=None)
    fake_df.columns = ['pt1', 'y1', 'phi1', 'm1', 'pt2', 'y2', 'phi2', 'm2']
    
    # Convert fake data to Cartesian for plotting
    fake_converted = {}
    for p_type in ['1', '2']:
        pt  = np.abs(fake_df[f'pt{p_type}'].values)  # pt must be positive
        y   = fake_df[f'y{p_type}'].values           # NO clipping — clipping was causing spikes!
        phi = fake_df[f'phi{p_type}'].values
        m   = fake_df[f'm{p_type}'].values

        mT = np.sqrt(pt**2 + m**2)
        pz = mT * np.sinh(y)
        E  = mT * np.cosh(y)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        
        fake_converted[f'pt{p_type}'] = pt
        fake_converted[f'y{p_type}']  = y
        fake_converted[f'phi{p_type}'] = phi
        fake_converted[f'm{p_type}']  = m
        fake_converted[f'px{p_type}'] = px
        fake_converted[f'py{p_type}'] = py
        fake_converted[f'pz{p_type}'] = pz
        fake_converted[f'E{p_type}']  = E
    fake_df = pd.DataFrame(fake_converted)
    
    # Drop any rows with NaN or Inf to prevent FPD crashes
    fake_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fake_df.dropna(inplace=True)
    

    # Map column names to LaTeX labels (track1=Kaon, track2=Pion)
    latex_labels = {
        'pt1': r'$p_{T, K}$ [GeV]', 'y1': r'$y_K$', 'phi1': r'$\phi_K$', 'm1': r'$m_K$ [GeV]',
        'pt2': r'$p_{T, \pi}$ [GeV]', 'y2': r'$y_\pi$', 'phi2': r'$\phi_\pi$', 'm2': r'$m_\pi$ [GeV]',
        'px1': r'$p_{x, K}$ [GeV]', 'py1': r'$p_{y, K}$ [GeV]', 'pz1': r'$p_{z, K}$ [GeV]', 'E1': r'$E_K$ [GeV]',
        'px2': r'$p_{x, \pi}$ [GeV]', 'py2': r'$p_{y, \pi}$ [GeV]', 'pz2': r'$p_{z, \pi}$ [GeV]', 'E2': r'$E_\pi$ [GeV]'
    }


    # Plot marginals (Physical)
    features_phys = ['pt1', 'y1', 'phi1', 'm1', 'pt2', 'y2', 'phi2', 'm2']
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    for i, feature in enumerate(features_phys):
        min_val = min(real_df[feature].min(), fake_df[feature].min())
        max_val = max(real_df[feature].max(), fake_df[feature].max())
        bins = np.linspace(min_val, max_val, 100)

        if(feature in ['pt1', 'pt2']):
            axs[i].set_xlim(0, 40)
            bins = np.linspace(min_val, max_val, 500)
        
        axs[i].hist(real_df[feature], bins=bins, alpha=0.9, label='Real', density=True, color='blue', histtype='step', lw=2)
        axs[i].hist(fake_df[feature], bins=bins, alpha=0.9, label='Generated', density=True, color='red', histtype='step', lw=2)
        axs[i].set_title(latex_labels[feature], fontsize=14)
        axs[i].legend()
        
    prefix = args.prefix if hasattr(args, 'prefix') and args.prefix else ''
    phys_plot = f'{prefix}marginals_comparison_physical.png'
    plt.tight_layout()
    plt.savefig(phys_plot)
    print(f"Saved physical marginals plot to {phys_plot}")
    plt.close()
    
    # Plot marginals (Cartesian)
    features_cart = ['px1', 'py1', 'pz1', 'E1', 'px2', 'py2', 'pz2', 'E2']
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    bins = np.linspace(-50, 50, 700)
    
    for i, feature in enumerate(features_cart):

        if(feature in ['E1', 'E2']):
            axs[i].set_xlim(0, 40)
        else:
            axs[i].set_xlim(-5, 5)

        axs[i].hist(real_df[feature], bins=bins, alpha=0.9, label='Real', density=True, color='blue', histtype='step', lw=2)
        axs[i].hist(fake_df[feature], bins=bins, alpha=0.9, label='Generated', density=True, color='red', histtype='step', lw=2)
        axs[i].set_title(latex_labels[feature], fontsize=14)
        axs[i].legend()
        
    
    cart_plot = f'{prefix}marginals_comparison_cartesian.png'
    plt.tight_layout()
    plt.savefig(cart_plot)
    print(f"Saved cartesian marginals plot to {cart_plot}")
    plt.close()
    
    # Invariant mass calculation (uses already-clipped values from fake_df)
    def calc_invariant_mass(df):
        px1, py1 = df['px1'].values, df['py1'].values
        pz1, E1  = df['pz1'].values, df['E1'].values
        px2, py2 = df['px2'].values, df['py2'].values
        pz2, E2  = df['pz2'].values, df['E2'].values
        
        E_tot  = E1  + E2
        px_tot = px1 + px2
        py_tot = py1 + py2
        pz_tot = pz1 + pz2
        
        mass_sq = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
        mass = np.sqrt(np.maximum(mass_sq, 0))
        return mass
        
    real_mass = calc_invariant_mass(real_df)
    fake_mass = calc_invariant_mass(fake_df)
    
    plt.figure(figsize=(8, 6))
    bins = np.linspace(1.7, 2.0, 50) # D0 mass is around 1.864 GeV
    plt.hist(real_mass, bins=bins, alpha=0.9, label='Real D0', density=True, color='blue', histtype='step', lw=2)
    plt.hist(fake_mass, bins=bins, alpha=0.9, label='Generated D0', density=True, color='red', histtype='step', lw=2)
    plt.xlabel(r'Invariant Mass $m(K\pi)$ [GeV]', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(r'$D^0 \to K^- \pi^+$ Invariant Mass', fontsize=14)
    plt.legend()
    mass_plot = f'{prefix}invariant_mass_comparison.png'
    plt.savefig(mass_plot)
    print(f"Saved invariant mass plot to {mass_plot}")
    plt.close()

    # --- FPD and KPD Calculation (Kansal et al. 2023) ---
    print("\nCalculating FPD  using Energy Flow Polynomials JETNET")
    
    # #  Convert to jetnet format (usually expects pt, eta, phi, mass)
    # #  We will use pt, y, phi, m

    def to_jetnet_format(df):
        N = len(df)
        events = np.zeros((N, 2, 4))
        for i, p_type in enumerate(['1', '2']): # K and Pi
            events[:, i, 0] = df[f'pt{p_type}'].values
            events[:, i, 1] = df[f'y{p_type}'].values
            events[:, i, 2] = df[f'phi{p_type}'].values
            events[:, i, 3] = df[f'm{p_type}'].values
        return events

    # Filter out any unphysical events before FPD calculation (e.g. from baseline models)
    valid_mask = (fake_df['pt1'] > 1e-3) & (fake_df['pt2'] > 1e-3) & \
                 np.isfinite(fake_df['y1']) & np.isfinite(fake_df['y2']) & \
                 np.isfinite(fake_df['phi1']) & np.isfinite(fake_df['phi2']) & \
                 (fake_df['m1'] >= 0) & (fake_df['m2'] >= 0)
    num_invalid = (~valid_mask).sum()
    if num_invalid > 0:
        print(f"[Note] Removed {num_invalid} unphysical / NaN events before FPD calculation.")
        fake_df_for_fpd = fake_df[valid_mask]
    else:
        fake_df_for_fpd = fake_df

    events_real = to_jetnet_format(real_df)
    events_fake = to_jetnet_format(fake_df_for_fpd)

    # 1. Extract the recommended features for the FPD metric (efp_jobs=1 required on Windows to avoid spawn crashes)
    try:
        real_features = jetnet.evaluation.get_fpd_kpd_jet_features(events_real, efp_jobs=1)
        fake_features = jetnet.evaluation.get_fpd_kpd_jet_features(events_fake, efp_jobs=1)

        # 2. Calculate the FPD score (returns a tuple of value and error)
        FPDscore, FPDerror = jetnet.evaluation.fpd(real_features, fake_features) 

        print(f"-> Fréchet Physics Distance (FPD) (score ± error) in jetnet lib: {FPDscore:.5f} ± {FPDerror:.5f}") 
        print("(OBS: Lower is better. A score close to 0 means the generated angular/energy correlations perfectly match the real data)")
    except Exception as e:
        print(f"-> [Warning] Failed to calculate FPD: {e}")

    
    #import energyflow as ef
    # Compute EFPs up to degree 4
    # energyflow expects (pt, y, phi) per particle — shape (N_events, N_particles, 3)
    # batch_compute iterates over events automatically
    #efpset = ef.EFPSet('d<=4', measure='hadr', beta=1)
    #efps_real = efpset.batch_compute(events_real[:, :, :3])  # pt, y, phi only
    #efps_fake = efpset.batch_compute(events_fake[:, :, :3])
    
    # Now use jetnet fpd/kpd on the EFP arrays directly if possible, or compute Fréchet distance manually.
    # Actually jetnet.evaluation provides a direct way:
    #from scipy.linalg import sqrtm
    
    #def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    #    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    #    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    #    diff = mu1 - mu2
    #   covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    #    if not np.isfinite(covmean).all():
    #        offset = np.eye(sigma1.shape[0]) * eps
    #        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    #    if np.iscomplexobj(covmean):
    #        covmean = covmean.real
    #    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    #mu_r, sig_r = np.mean(efps_real, axis=0), np.cov(efps_real, rowvar=False)
    #mu_f, sig_f = np.mean(efps_fake, axis=0), np.cov(efps_fake, rowvar=False)
    #fpd_score = calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f)
    #print("-> Note: FPD is computed using Energy Flow Polynomials (EFPs) up to degree 4.")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_real = 'data_D0_test.txt' if os.path.exists('data_D0_test.txt') else 'data_D0_daughtersPx_Py_Pz_E_trk1_trk2.txt'
    parser.add_argument('--real_data', type=str, default=default_real, help='Path to real dataset (defaults to data_D0_test.txt if present)')
    parser.add_argument('--fake_data', type=str, default='generated_samples.txt', help='Path to generated samples')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for output plot filenames (e.g. cartesian_standard_)')
    args = parser.parse_args()
    
    evaluate(args)
