"""
Load Fink/LSST alerts (ftransfer parquet) in the format expected by astro-mtan.

Replaces `read_alert` and the ZTF-specific cuts of main_preprocessing.py.
The returned DataFrame has the columns that get_lc / prepare_data expect:
    objectId, jd, fid, magpsf, sigmapsf, finkclass, tnsclass
with fid = 1, 2, 3, 4 for g, r, i, z.

Columns read from the parquet:
    diaObjectId, diaSourceId, midpointMjdTai, band, psfFlux, psfFluxErr,
    scienceFlux, scienceFluxErr
"""
import numpy as np
import pandas as pd

BANDS = ["g", "r", "i", "z"]                          # fixed order = channels 0..3
BAND2FID = {b: k + 1 for k, b in enumerate(BANDS)}    # g=1 r=2 i=3 z=4
NJY_AB_ZP = 31.4                                      # mag_AB = 31.4 - 2.5 log10(flux_nJy)


def load_lsst_alerts(path, flux_mode="diff", snr_min=5.0):
    """
    flux_mode:
      'diff'    -> psfFlux (difference image: variation w.r.t. the template, host subtracted).
                   ZTF analogue: magpsf with isdiffpos == 't' (SN case of the paper).
      'science' -> scienceFlux (total flux on the science image, host included).
                   Analogue of the lc_correction used for AGN.
    """
    # dtype_backend="pyarrow": diaObjectId values (~3e17 > 2**53) must stay 64-bit integers.
    # With null values, the numpy backend would cast them to float64 and silently merge distinct objects.
    df = pd.read_parquet(path, dtype_backend="pyarrow")
    if pd.api.types.is_float_dtype(df["diaObjectId"]):
        raise ValueError("diaObjectId was read as float: precision loss on the identifiers.")
    num = ["midpointMjdTai", "psfFlux", "psfFluxErr", "scienceFlux", "scienceFluxErr"]
    df[num] = df[num].astype("float64")

    n0 = len(df)
    df = df.dropna(subset=["diaObjectId"])             # alerts without an object: no light curve possible
    df = df.drop_duplicates("diaSourceId")
    df = df[df["band"].isin(BANDS)]                    # drop u and y

    if flux_mode == "diff":
        flux, err = df["psfFlux"], df["psfFluxErr"]
    elif flux_mode == "science":
        flux, err = df["scienceFlux"], df["scienceFluxErr"]
    else:
        raise ValueError(flux_mode)

    keep = (flux > 0) & (err > 0) & (flux / err > snr_min)   # also guarantees flux > 0 for the log
    df, flux, err = df[keep], flux[keep], err[keep]

    out = pd.DataFrame({
        "objectId": df["diaObjectId"].astype("int64").astype(str).values,
        "jd": df["midpointMjdTai"].values,             # MJD in days: only time differences matter
        "fid": df["band"].map(BAND2FID).astype(int).values,
        "magpsf": NJY_AB_ZP - 2.5 * np.log10(flux.values),
        "sigmapsf": 1.0857 * err.values / flux.values,
        # get_lc / prepare_data read these label columns; without Fink classes we use a single label.
        "finkclass": "lsst",
        "tnsclass": "Unknown",
    })
    out = out.sort_values(["objectId", "jd"]).reset_index(drop=True)
    print(f"{n0} rows -> {len(out)} detections kept "
          f"({out.objectId.nunique()} objects, flux_mode={flux_mode}, snr>{snr_min})")
    return out


def count_nights(jd):
    """Number of distinct observing nights. A Chilean night straddles 00:00 UTC, so nights are split
    at ~15:30 UTC (local noon): MJD fraction 0.65."""
    return np.unique(np.floor(np.asarray(jd) - 0.65)).size


def object_stats(df):
    """Per-object statistics, computed once with vectorized groupby operations (no Python loop).
    Columns: n_points, t_min, t_max, duration_d, g, r, i, z (points per band), n_nights."""
    g = df.groupby("objectId", sort=False)
    st = pd.DataFrame({"n_points": g.size(), "t_min": g["jd"].min(), "t_max": g["jd"].max()})
    st["duration_d"] = st["t_max"] - st["t_min"]
    per_band = (df.groupby(["objectId", "fid"], sort=False).size().unstack(fill_value=0)
                  .reindex(columns=[1, 2, 3, 4], fill_value=0))
    per_band.columns = BANDS
    st = st.join(per_band)
    # same night definition as count_nights: floor(MJD - 0.65)
    night = pd.DataFrame({"objectId": df["objectId"].values,
                          "night": np.floor(df["jd"].values - 0.65)}).drop_duplicates()
    st["n_nights"] = night.groupby("objectId", sort=False).size()
    return st


def apply_cuts(st, min_total=10, min_per_band=3, min_bands=2,
               min_duration_days=None, min_nights=None):
    """Boolean mask (indexed by objectId) of the objects passing the cuts, from object_stats()."""
    keep = (st["n_points"] >= min_total) & ((st[BANDS] >= min_per_band).sum(axis=1) >= min_bands)
    if min_duration_days is not None:
        keep &= st["duration_d"] >= min_duration_days
    if min_nights is not None:
        keep &= st["n_nights"] >= min_nights
    return keep


def select_objects(df, min_total=10, min_per_band=3, min_bands=2,
                   min_duration_days=None, min_nights=None):
    """
    Vectorized version (same result as select_objects_reference, orders of magnitude faster).
    Replaces the ZTF cut '>= 3 points in each band' (g AND r).
    An object is kept if it has >= min_total points, >= min_per_band points in at least min_bands bands,
    and (optional) a time span >= min_duration_days and >= min_nights distinct nights.
    Setting min_per_band=1, min_bands=1 reproduces the notebook cut (> 10 alerts, any band).
    Note: min_per_band must be >= 1.
    """
    if min_per_band < 1:
        raise ValueError("min_per_band must be >= 1")
    st = object_stats(df)
    keep_ids = st.index[apply_cuts(st, min_total, min_per_band, min_bands,
                                   min_duration_days, min_nights)]
    out = df[df["objectId"].isin(keep_ids)]
    print(f"{out.objectId.nunique()} / {len(st)} objects kept "
          f"(>= {min_total} points, >= {min_per_band} points in >= {min_bands} bands, "
          f"duration >= {min_duration_days} d, nights >= {min_nights})")
    return out


def select_objects_reference(df, min_total=10, min_per_band=3, min_bands=2,
                   min_duration_days=None, min_nights=None):
    """
    SLOW reference implementation (Python loop over objects), kept only to test select_objects.
    Replaces the ZTF cut '>= 3 points in each band' (g AND r).
    An object is kept if it has >= min_total points, >= min_per_band points in at least min_bands bands,
    and (optional) a time span >= min_duration_days and >= min_nights distinct nights.
    Setting min_per_band=1, min_bands=1 reproduces the notebook cut (> 10 alerts, any band).
    """
    def ok(g):
        per_band = g["fid"].value_counts()
        if len(g) < min_total or (per_band >= min_per_band).sum() < min_bands:
            return False
        if min_duration_days is not None and g["jd"].max() - g["jd"].min() < min_duration_days:
            return False
        if min_nights is not None and count_nights(g["jd"]) < min_nights:
            return False
        return True
    out = df.groupby("objectId").filter(ok)
    print(f"{out.objectId.nunique()} / {df.objectId.nunique()} objects kept "
          f"(>= {min_total} points, >= {min_per_band} points in >= {min_bands} bands, "
          f"duration >= {min_duration_days} d, nights >= {min_nights})")
    return out


def _self_test(n_obj=3000, seed=0):
    """Check that the vectorized selection equals the reference one on synthetic light curves."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_obj):
        n = int(rng.integers(1, 40))
        t0 = 61000 + rng.uniform(0, 100)
        span = rng.choice([0.05, 0.5, 3, 30])
        jd = t0 + rng.uniform(0, span, n)
        rows.append(pd.DataFrame({"objectId": str(170000000000000000 + k), "jd": jd,
                                  "fid": rng.choice([1, 2, 3, 4], n, p=[.4, .4, .1, .1])}))
    df = pd.concat(rows, ignore_index=True)
    for params in [dict(), dict(min_nights=3), dict(min_per_band=1, min_bands=1),
                   dict(min_per_band=5, min_bands=2, min_nights=2, min_duration_days=1.0),
                   dict(min_per_band=3, min_bands=3, min_nights=5)]:
        a = set(select_objects(df, **params)["objectId"])
        b = set(select_objects_reference(df, **params)["objectId"])
        assert a == b, f"MISMATCH for {params}: {len(a ^ b)} differing objects"
        print("  OK", params, f"({len(a)} objects)")
    print("self-test passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _self_test()
        sys.exit(0)
    path = sys.argv[1] if len(sys.argv) > 1 else "df_50first_lines.parquet"
    a = load_lsst_alerts(path)
    print(a.head())
