"""Smoke tests of the weighted-likelihood branch ("fusion"). Needs torch; CPU only, about a minute.

    python test_fusion_weights.py

1. unit tests of the weighted loss / MSE (mtan_utils);
2. prepare_data with and without weights: same split, same data/mask/time, weight > 0 <=> mask == 1;
3. truncation transform keeps the weight channel consistent;
4. end to end on a synthetic Fink/LSST transfer: make_parquet -> preprocessing -> 2 training epochs
   (weights on, --no-use-weights, and data without weight channel).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mtan_utils                                        # noqa: E402
import prepare_lsst as P                                 # noqa: E402
from prepare_data_lsst import ContinuousTruncateLightCurve, prepare_data   # noqa: E402

DIM = 4


def make_ftransfer(dirpath, n_obj=60, seed=1):
    """Synthetic ftransfer-like parquet: several detections per night and band, 8 nights, 4 bands."""
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for k in range(n_obj):
        oid = 317000000000000000 + k
        base = rng.uniform(21, 23)
        for night in range(8):
            for band in rng.choice(['g', 'r', 'i', 'z'], 2, replace=False):
                for rep in range(int(rng.integers(1, 6))):
                    mag = base + 0.3 * np.sin(night / 2 + k) + rng.normal(0, 0.05)
                    flux = 10 ** ((31.4 - mag) / 2.5)
                    err = flux / rng.uniform(6, 30)
                    rows.append(dict(diaObjectId=oid, diaSourceId=sid, midpointMjdTai=61000 + night * 3 + 0.8 + 0.05 * rep,
                                     band=band, psfFlux=flux, psfFluxErr=err, scienceFlux=flux, scienceFluxErr=err))
                    sid += 1
    os.makedirs(dirpath, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(dirpath, 'part0.parquet'))


def test_losses():
    torch.manual_seed(0)
    B, T, K, NREF, L = 3, 7, 2, 5, 4
    mask = (torch.rand(B, T, DIM) > 0.5).float()
    data = torch.randn(B, T, DIM) * mask
    tt = torch.rand(B, T).unsqueeze(-1)
    w = (torch.rand(B, T, DIM) + 0.5) * mask
    pred = torch.randn(K, B, T, DIM)
    qm, ql = torch.randn(B, NREF, L), torch.randn(B, NREF, L)
    on = SimpleNamespace(std=0.01, norm=False, use_weights=True)
    off = SimpleNamespace(std=0.01, norm=False, use_weights=False)
    cpu = torch.device('cpu')

    b_u = torch.cat([data, mask, tt], 2)                  # no weight channel
    b_w = torch.cat([data, mask, w, tt], 2)
    b_1 = torch.cat([data, mask, mask, tt], 2)            # weights all 1 where observed
    b_2 = torch.cat([data, mask, 2 * mask, tt], 2)
    lu, klu = mtan_utils.compute_losses(DIM, b_u, qm, ql, pred, on, cpu)
    lw, klw = mtan_utils.compute_losses(DIM, b_w, qm, ql, pred, on, cpu)
    lw_off, _ = mtan_utils.compute_losses(DIM, b_w, qm, ql, pred, off, cpu)
    l1, _ = mtan_utils.compute_losses(DIM, b_1, qm, ql, pred, on, cpu)
    l2, _ = mtan_utils.compute_losses(DIM, b_2, qm, ql, pred, on, cpu)
    assert torch.allclose(lw_off, lu), 'weights must be ignored when use_weights is False'
    assert torch.allclose(l1, lu), 'unit weights must reproduce the unweighted loss'
    assert torch.allclose(l2, 2 * lu), 'doubling the weights must double the data-fit term'
    assert torch.equal(klw, klu), 'the KL term must not depend on the weights'
    assert not torch.allclose(lw, lu)

    m_u = mtan_utils.mean_squared_error(data, pred[0], mask)
    assert torch.allclose(mtan_utils.mean_squared_error(data, pred[0], mask, weight=mask), m_u)
    manual = (((data - pred[0]) ** 2) * mask * w).sum() / (mask * w).sum()
    assert torch.allclose(mtan_utils.mean_squared_error(data, pred[0], mask, weight=w), manual)
    print('ok test_losses')


def test_prepare_data_and_truncation(df):
    dfw = df.assign(weight=P.compute_weights(df, 'invvar'))
    o_w = prepare_data(dfw, dim=DIM, train_batch_size=4, convert_to_tensor=True, weight_column='weight')
    o_u = prepare_data(df, dim=DIM, train_batch_size=4, convert_to_tensor=True)
    for split in ('train', 'val', 'test'):
        assert list(o_w[f'{split}_objIds']) == list(o_u[f'{split}_objIds']), 'the split must not depend on weights'
        a, b = o_w[f'{split}_data_combined'], o_u[f'{split}_data_combined']
        assert a.shape[-1] == 3 * DIM + 1 and b.shape[-1] == 2 * DIM + 1
        assert torch.equal(a[..., :2 * DIM], b[..., :2 * DIM]) and torch.equal(a[..., -1], b[..., -1])
        assert torch.equal(a[..., 2 * DIM:3 * DIM] > 0, a[..., DIM:2 * DIM] > 0), 'weight > 0 <=> mask == 1'
    print('ok test_prepare_data')

    x = o_w['train_data_combined'][0]
    tr = ContinuousTruncateLightCurve((0.3, 0.7), dim=DIM, min_datapoints_each_filter=1)
    np.random.seed(0)
    n_cut = 0
    for _ in range(20):
        y = tr(x.clone())
        assert y.shape == x.shape
        assert torch.equal(y[:, 2 * DIM:3 * DIM] > 0, y[:, DIM:2 * DIM] > 0)
        assert bool((y[:, DIM:2 * DIM] <= x[:, DIM:2 * DIM]).all())
        assert torch.equal(y[:, -1], x[:, -1])
        n_cut += int(y[:, DIM:2 * DIM].sum() < x[:, DIM:2 * DIM].sum())
    assert n_cut > 0, 'the truncation never removed a point: test is vacuous'
    print('ok test_truncation')


def run(script, args, cwd):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONPATH': HERE})
    assert r.returncode == 0, f'{script} failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}'
    return r.stdout


def test_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        topic = os.path.join(tmp, 'ftransfer_lsst_test')
        make_ftransfer(topic)
        run('make_parquet_lsst.py', [topic], tmp)
        pq = 'alerts_processed_ftransfer_lsst_test.parquet'
        run('main_preprocessing_lsst_from_parquet.py', [pq, '--weight', 'invvar'], tmp)
        run('main_preprocessing_lsst_from_parquet.py', [pq, '--weight', 'none'], tmp)

        def train(tag, *extra):
            return run('tan_unsupervised_lsst.py',
                       ['--dim', '4', '--latent-dim', '4', '--rec-hidden', '16', '--gen-hidden', '16',
                        '--embed-time', '16', '--enc', 'mtan_rnn', '--dec', 'mtan_rnn', '--kl', '--learn-emb',
                        '--ref-resolution-days', '2', '--niters', '2', '--dataset', f'lsst_{tag}',
                        '--train_val_test_min_max_times_filename', f'./train_val_test_min_max_times_{tag}.npy',
                        *extra], tmp)

        out = train('ftransfer_lsst_test_winvvar')
        assert 'Weighted loss and metrics: True' in out and 'val_mse_unw' in out, out[-2000:]
        out = train('ftransfer_lsst_test_winvvar', '--no-use-weights')
        assert 'Weighted loss and metrics: False' in out and 'val_mse_unw' not in out, out[-2000:]
        out = train('ftransfer_lsst_test')                 # no weight channel
        assert 'WARNING: --use-weights requested' in out and 'val_mse_unw' not in out, out[-2000:]
        print('ok test_end_to_end')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_losses()
    tmp = tempfile.mkdtemp()
    try:
        make_ftransfer(os.path.join(tmp, 'ftransfer_lsst_unit'))
        frame = P.select_objects(P.load_lsst_alerts(os.path.join(tmp, 'ftransfer_lsst_unit')),
                                 min_total=10, min_per_band=3, min_bands=2, min_nights=3)
        test_prepare_data_and_truncation(frame)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    test_end_to_end()
    print('all tests passed')
