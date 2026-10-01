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


def select_objects(df, min_total=10, min_per_band=3, min_bands=2,
                   min_duration_days=None, min_nights=None):
    """
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


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "df_50first_lines.parquet"
    a = load_lsst_alerts(path)
    print(a.head())
