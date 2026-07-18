import pandas as pd
import numpy as np

df = pd.read_csv('data_D0_daughtersPx_Py_Pz_E_trk1_trk2.txt', sep=r'\s+', header=None)
df.columns = ['px1', 'px2', 'py1', 'py2', 'pz1', 'pz2', 'E1', 'E2']

# Compute invariant mass for each track from E^2 - p^2
p1 = np.sqrt(df['px1']**2 + df['py1']**2 + df['pz1']**2)
m1 = np.sqrt(np.maximum(df['E1']**2 - p1**2, 0))

p2 = np.sqrt(df['px2']**2 + df['py2']**2 + df['pz2']**2)
m2 = np.sqrt(np.maximum(df['E2']**2 - p2**2, 0))

print(f'Track 1 mass: mean={m1.mean():.6f} GeV  std={m1.std():.6f} GeV  median={m1.median():.6f} GeV')
print(f'Track 2 mass: mean={m2.mean():.6f} GeV  std={m2.std():.6f} GeV  median={m2.median():.6f} GeV')
print()
print(f'Pion mass  = 0.139570 GeV')
print(f'Kaon mass  = 0.493677 GeV')
print()
print('First 10 rows m1:', m1.values[:10].round(6))
print('First 10 rows m2:', m2.values[:10].round(6))
