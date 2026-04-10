"""Cross-phase 30-model blend optimizer."""
import pickle
import numpy as np
import pandas as pd
import os
import glob
from scipy.optimize import minimize

CKPT_DIR = 'output'

def load_phase_oofs(phase_pattern='ckpt_phase*_*.pkl'):
    """모든 Phase OOF 로드."""
    oofs = {}
    tests = {}
    for path in sorted(glob.glob(os.path.join(CKPT_DIR, phase_pattern))):
        name = os.path.basename(path).replace('.pkl', '')
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        if 'oof' in ckpt and 'test' in ckpt:
            oofs[name] = ckpt['oof']
            tests[name] = ckpt['test']
            print(f"  Loaded: {name} (CV {np.abs(ckpt.get('cv_mae', 0)):.4f})")
    return oofs, tests

def optimize_blend(oofs, y_train):
    """Nelder-Mead normalized blend."""
    names = list(oofs.keys())
    oof_matrix = np.column_stack([oofs[n] for n in names])

    def objective(w, oof_m, y):
        w = w / (w.sum() + 1e-12)
        return np.abs((oof_m * w).sum(axis=1) - y).mean()

    x0 = np.ones(len(names)) / len(names)
    result = minimize(objective, x0, args=(oof_matrix, y_train),
                      method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    weights = result.x / (result.x.sum() + 1e-12)
    cv = np.abs((oof_matrix * weights).sum(axis=1) - y_train).mean()

    print(f"\nBlend CV: {cv:.4f}")
    print(f"Weights:")
    for n, w in sorted(zip(names, weights), key=lambda x: -abs(x[1])):
        if abs(w) > 0.001:
            print(f"  {n}: {w:.4f}")

    return weights, names, cv

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    y_train = train['avg_delay_minutes_next_30m'].values

    oofs, tests = load_phase_oofs()
    print(f"\nTotal models: {len(oofs)}")

    if len(oofs) >= 2:
        weights, names, cv = optimize_blend(oofs, y_train)
