#!/usr/bin/env python3
"""Preprocess raw interferometer ASC data for stitching.

Loads raw ASC maps, selects/crops the effective aperture area, optionally
subtracts a calibrated interferometer systematic, removes per-frame low-order
piston/tilt/curvature terms, optionally refines raster positions by phase
correlation, and writes both TIFF stacks and a processed NPZ consumed by
stitching_interferometer_autodiff.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from typing import Any

import numpy as np

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def list_input_files(raw_dir: Path, pattern: str) -> list[Path]:
    files = sorted(raw_dir.glob(pattern), key=natural_sort_key)
    if files:
        return files
    literal_dir = raw_dir
    if not literal_dir.exists() and raw_dir.parent.exists():
        for child in raw_dir.parent.iterdir():
            if child.name == raw_dir.name:
                literal_dir = child
                break
    if literal_dir.exists() and literal_dir.is_dir() and "/" not in pattern and "\\" not in pattern:
        return sorted(
            [path for path in literal_dir.iterdir() if fnmatch.fnmatch(path.name, pattern)],
            key=natural_sort_key,
        )
    return files


def load_asc_height_map(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    lines = path.read_text(errors="replace").splitlines()
    blank = next(i for i, line in enumerate(lines) if line.strip() == "")
    meta: dict[str, Any] = {}
    for line in lines[:blank]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        value: Any = parts[1].strip()
        unit = parts[2].strip() if len(parts) >= 3 else ""
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass
        meta[key] = value
        if unit:
            meta[f"{key} Unit"] = unit
    z = np.genfromtxt(path, delimiter="\t", skip_header=blank + 1, dtype=np.float64, invalid_raise=False, autostrip=True)
    if z.ndim == 2 and z.shape[1] > 0 and np.all(~np.isfinite(z[:, -1])):
        z = z[:, :-1]
    z[~np.isfinite(z)] = np.nan
    return z, meta


def load_asc_worker(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    return load_asc_height_map(Path(path))


def load_raw_stack(files: list[Path], cpu_cores: int) -> tuple[np.ndarray, dict[str, Any]]:
    workers = max(1, int(cpu_cores))
    if workers == 1 or len(files) <= 1:
        loaded = [load_asc_worker(str(path)) for path in tqdm(files, desc="Loading raw frames")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            loaded = list(tqdm(executor.map(load_asc_worker, [str(path) for path in files]), total=len(files), desc=f"Loading raw frames ({workers} cores)"))
    frames = [item[0] for item in loaded]
    first_meta = loaded[0][1] if loaded else {}
    return np.asarray(frames, dtype=np.float64), first_meta


def parse_crop(crop: str | None) -> tuple[slice, slice] | None:
    if not crop:
        return None
    parts = [int(part) if part else None for part in crop.split(":")]
    if len(parts) != 4:
        raise ValueError("crop must be y0:y1:x0:x1")
    return slice(parts[0], parts[1]), slice(parts[2], parts[3])


def crop_to_tuple(crop: tuple[slice, slice], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = crop
    y0, y1, _ = ys.indices(shape[0])
    x0, x1, _ = xs.indices(shape[1])
    return y0, y1, x0, x1


def auto_crop_from_finite(stack: np.ndarray, margin: int, min_fraction: float) -> tuple[slice, slice]:
    finite_fraction = np.mean(np.isfinite(stack), axis=0)
    valid = finite_fraction >= min_fraction
    if not np.any(valid):
        raise ValueError("auto crop found no pixels with enough finite coverage")
    yy, xx = np.where(valid)
    y0 = max(int(yy.min()) - margin, 0)
    y1 = min(int(yy.max()) + margin + 1, stack.shape[1])
    x0 = max(int(xx.min()) - margin, 0)
    x1 = min(int(xx.max()) + margin + 1, stack.shape[2])
    return slice(y0, y1), slice(x0, x1)


def nanmedian_no_warning(stack: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(stack)
    filled = np.where(finite, stack, np.inf)
    count = finite.sum(axis=axis)
    kth = np.maximum((count - 1) // 2, 0)
    sorted_values = np.sort(filled, axis=axis)
    median = np.take_along_axis(sorted_values, np.expand_dims(kth, axis=axis), axis=axis).squeeze(axis)
    median[count == 0] = np.nan
    return median


def manual_crop_gui(image: np.ndarray) -> tuple[slice, slice]:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    fig, ax = plt.subplots(figsize=(9, 7))
    vals = image[np.isfinite(image)]
    if vals.size:
        lo, hi = np.nanpercentile(vals, [1, 99])
    else:
        lo, hi = 0.0, 1.0
    ax.imshow(image, origin="upper", cmap="viridis", vmin=lo, vmax=hi)
    ax.set_title("Drag crop rectangle, then press Enter")
    selected: dict[str, tuple[int, int, int, int]] = {}

    def on_select(eclick, erelease):
        x0, x1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        y0, y1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
        selected["crop"] = (max(y0, 0), min(y1 + 1, image.shape[0]), max(x0, 0), min(x1 + 1, image.shape[1]))

    selector = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, interactive=True)

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    selector.set_active(False)
    if "crop" not in selected:
        raise RuntimeError("No crop selected")
    y0, y1, x0, x1 = selected["crop"]
    return slice(y0, y1), slice(x0, x1)


def downsample_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return stack
    return stack[:, ::factor, ::factor]


def load_calibration(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".asc":
        z, _ = load_asc_height_map(p)
        return z
    if suffix == ".npy":
        return np.asarray(np.load(p, allow_pickle=False), dtype=np.float64)
    if suffix == ".npz":
        with np.load(p, allow_pickle=True) as z:
            for key in ("systematic", "systematic_true", "calibration", "image", "arr_0"):
                if key in z:
                    arr = z[key]
                    if arr.size:
                        return np.asarray(arr, dtype=np.float64)
        raise KeyError(f"No calibration-like array found in {p}")
    try:
        import tifffile
        arr = tifffile.imread(p)
    except Exception:
        import imageio.v3 as iio
        arr = iio.imread(p)
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float64)


def low_order_bases(shape: tuple[int, int], order: str, scan_axis: str) -> list[np.ndarray]:
    h, w = shape
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    bases = [np.ones(shape, dtype=np.float64)]
    if order == "tilt":
        bases += [xx, yy]
    return bases


def as_2d_calibration(calibration: np.ndarray) -> np.ndarray:
    arr = np.asarray(calibration, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = np.nanmedian(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Calibration data must reduce to a 2D image, got shape {arr.shape}")
    arr[~np.isfinite(arr)] = np.nan
    return arr


def fit_low_order(surface: np.ndarray, mask: np.ndarray, order: str, scan_axis: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(surface) & mask
    if not np.any(valid) or order == "none":
        return np.zeros_like(surface), np.zeros(0)
    bases = low_order_bases(surface.shape, order, scan_axis)
    A = np.stack([b[valid] for b in bases], axis=1)
    coeff, *_ = np.linalg.lstsq(A, surface[valid], rcond=None)
    trend = sum(c * b for c, b in zip(coeff, bases))
    return trend, coeff



def remove_low_order_frame_worker(payload: tuple[np.ndarray, np.ndarray, str, str]) -> tuple[np.ndarray, np.ndarray]:
    frame, mask, fit_order, scan_axis = payload
    trend, coeff = fit_low_order(frame, mask > 0, fit_order, scan_axis)
    out = frame.copy()
    valid = np.isfinite(out)
    out[valid] = out[valid] - trend[valid]
    return out, coeff


def remove_low_order_stack(stack: np.ndarray, masks: np.ndarray, order: str, scan_axis: str, cpu_cores: int) -> tuple[np.ndarray, np.ndarray]:
    if order == "none":
        return stack.copy(), np.zeros((stack.shape[0], 0), dtype=np.float64)
    payloads = [(stack[i], masks[i], order, scan_axis) for i in range(stack.shape[0])]
    workers = max(1, int(cpu_cores))
    if workers == 1 or len(payloads) <= 1:
        results = [remove_low_order_frame_worker(payload) for payload in tqdm(payloads, desc="Removing piston/tilt")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(remove_low_order_frame_worker, payloads), total=len(payloads), desc=f"Removing piston/tilt ({workers} cores)"))
    out = np.asarray([result[0] for result in results], dtype=np.float64)
    coeffs = np.asarray([result[1] for result in results], dtype=np.float64)
    return out, coeffs


def physical_coordinates_mm(shape: tuple[int, int], pixel_spacing_y_mm: float, pixel_spacing_x_mm: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * pixel_spacing_y_mm
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * pixel_spacing_x_mm
    return np.meshgrid(y, x, indexing="ij")


def curvature_bases(
    shape: tuple[int, int],
    mode: str,
    pixel_spacing_y_mm: float,
    pixel_spacing_x_mm: float,
) -> tuple[list[np.ndarray], list[str]]:
    yy_mm, xx_mm = physical_coordinates_mm(shape, pixel_spacing_y_mm, pixel_spacing_x_mm)
    if mode == "1dx":
        return [xx_mm**2], ["x2_nm_per_mm2"]
    if mode == "1dy":
        return [yy_mm**2], ["y2_nm_per_mm2"]
    if mode == "2d":
        return [xx_mm**2, xx_mm * yy_mm, yy_mm**2], ["x2_nm_per_mm2", "xy_nm_per_mm2", "y2_nm_per_mm2"]
    raise ValueError(f"Unknown curvature mode: {mode}")


def fit_curvature(
    surface: np.ndarray,
    mask: np.ndarray,
    mode: str,
    pixel_spacing_y_mm: float,
    pixel_spacing_x_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(surface) & mask
    quad_bases, _ = curvature_bases(surface.shape, mode, pixel_spacing_y_mm, pixel_spacing_x_mm)
    yy_mm, xx_mm = physical_coordinates_mm(surface.shape, pixel_spacing_y_mm, pixel_spacing_x_mm)
    nuisance_bases = [np.ones(surface.shape, dtype=np.float64), xx_mm, yy_mm]
    fit_bases = quad_bases + nuisance_bases
    if not np.any(valid):
        return np.zeros_like(surface), np.zeros(len(quad_bases), dtype=np.float64)
    A = np.stack([basis[valid] for basis in fit_bases], axis=1)
    coeff_all, *_ = np.linalg.lstsq(A, surface[valid], rcond=None)
    coeff = coeff_all[: len(quad_bases)]
    curvature_trend = sum(c * basis for c, basis in zip(coeff, quad_bases))
    return curvature_trend, coeff


def remove_curvature_frame_worker(payload: tuple[np.ndarray, np.ndarray, str, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame, mask, mode, pixel_spacing_y_mm, pixel_spacing_x_mm = payload
    trend, coeff = fit_curvature(frame, mask > 0, mode, pixel_spacing_y_mm, pixel_spacing_x_mm)
    out = frame.copy()
    valid = np.isfinite(out)
    out[valid] = out[valid] - trend[valid]
    trend_masked = np.where(valid, trend, np.nan)
    return out, coeff, trend_masked


def finite_rms(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.sqrt(np.mean(values[valid] ** 2))) if np.any(valid) else float("nan")


def finite_pv(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.nanmax(values[valid]) - np.nanmin(values[valid])) if np.any(valid) else float("nan")


def curvature_statistics(coeffs: np.ndarray, trends: np.ndarray, mode: str, pixel_spacing_y_mm: float, pixel_spacing_x_mm: float) -> dict[str, Any]:
    if mode == "none" or coeffs.size == 0:
        return {"mode": mode, "coefficient_names": [], "coefficient_unit": "nm/mm^2", "radius_names": [], "radius_unit": "m", "coefficient_mean": [], "coefficient_rms": [], "coefficient_pv": [], "radius_mean_m": [], "radius_rms_m": [], "radius_pv_m": [], "map_rms_mean_nm": float("nan"), "map_rms_rms_nm": float("nan"), "map_pv_mean_nm": float("nan"), "map_pv_rms_nm": float("nan"), "pixel_spacing_y_mm": pixel_spacing_y_mm, "pixel_spacing_x_mm": pixel_spacing_x_mm}
    _, names = curvature_bases(trends.shape[1:], mode, pixel_spacing_y_mm, pixel_spacing_x_mm)
    radius = np.full_like(coeffs, np.nan, dtype=np.float64)
    nonzero = np.abs(coeffs) > 1e-30
    radius[nonzero] = 1000.0 / coeffs[nonzero] / 2
    if mode == "1dx":
        radius_names = ["Rx_m"]
    elif mode == "1dy":
        radius_names = ["Ry_m"]
    else:
        radius_names = ["Rx_m", "Rxy_equivalent_m", "Ry_m"]
    frame_rms = np.asarray([finite_rms(trend) for trend in trends], dtype=np.float64)
    frame_pv = np.asarray([finite_pv(trend) for trend in trends], dtype=np.float64)
    return {
        "mode": mode,
        "coefficient_names": names,
        "coefficient_unit": "nm/mm^2",
        "radius_names": radius_names,
        "radius_unit": "m",
        "pixel_spacing_y_mm": pixel_spacing_y_mm,
        "pixel_spacing_x_mm": pixel_spacing_x_mm,
        "coefficient_mean": np.mean(coeffs, axis=0).tolist(),
        "coefficient_rms": np.sqrt(np.mean(coeffs**2, axis=0)).tolist(),
        "coefficient_pv": (np.max(coeffs, axis=0) - np.min(coeffs, axis=0)).tolist(),
        "radius_mean_m": np.nanmean(radius, axis=0).tolist(),
        "radius_rms_m": np.nanstd(radius, axis=0).tolist(),
        "radius_pv_m": (np.nanmax(radius, axis=0) - np.nanmin(radius, axis=0)).tolist(),
        "map_rms_mean_nm": float(np.nanmean(frame_rms)),
        "map_rms_rms_nm": float(np.sqrt(np.nanmean(frame_rms**2))),
        "map_pv_mean_nm": float(np.nanmean(frame_pv)),
        "map_pv_rms_nm": float(np.sqrt(np.nanmean(frame_pv**2))),
    }


def remove_curvature_stack(stack: np.ndarray, masks: np.ndarray, mode: str, cpu_cores: int, pixel_spacing_y_mm: float, pixel_spacing_x_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if mode == "none":
        coeffs = np.zeros((stack.shape[0], 0), dtype=np.float64)
        trends = np.zeros_like(stack)
        return stack.copy(), coeffs, trends, curvature_statistics(coeffs, trends, mode, pixel_spacing_y_mm, pixel_spacing_x_mm)
    payloads = [(stack[i], masks[i], mode, pixel_spacing_y_mm, pixel_spacing_x_mm) for i in range(stack.shape[0])]
    workers = max(1, int(cpu_cores))
    if workers == 1 or len(payloads) <= 1:
        results = [remove_curvature_frame_worker(payload) for payload in tqdm(payloads, desc=f"Removing curvature {mode}")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(remove_curvature_frame_worker, payloads), total=len(payloads), desc=f"Removing curvature {mode} ({workers} cores)"))
    out = np.asarray([result[0] for result in results], dtype=np.float64)
    coeffs = np.asarray([result[1] for result in results], dtype=np.float64)
    trends = np.asarray([result[2] for result in results], dtype=np.float64)
    return out, coeffs, trends, curvature_statistics(coeffs, trends, mode, pixel_spacing_y_mm, pixel_spacing_x_mm)

def scan_positions_from_grid(n_frames: int, grid_y: int, grid_x: int, step_px: float, serpentine: bool) -> np.ndarray:
    if grid_y * grid_x != n_frames:
        raise ValueError(f"grid {grid_y}x{grid_x} does not match {n_frames} frames")
    positions = []
    for iy in range(grid_y):
        for ix_file in range(grid_x):
            ix = grid_x - 1 - ix_file if serpentine and iy % 2 else ix_file
            positions.append([iy * step_px, ix * step_px])
    return np.asarray(positions, dtype=np.float64)


def phase_correlation_shift(a: np.ndarray, b: np.ndarray, mask: np.ndarray, max_shift: int) -> tuple[float, float, float]:
    valid = np.isfinite(a) & np.isfinite(b) & mask
    if np.count_nonzero(valid) < 16:
        return 0.0, 0.0, 0.0
    aa = np.where(valid, a - np.nanmean(a[valid]), 0.0)
    bb = np.where(valid, b - np.nanmean(b[valid]), 0.0)
    fa = np.fft.fft2(aa)
    fb = np.fft.fft2(bb)
    cps = fa * np.conj(fb)
    cps /= np.maximum(np.abs(cps), 1e-12)
    corr = np.fft.fftshift(np.fft.ifft2(cps).real)
    cy, cx = np.array(corr.shape) // 2
    y0, y1 = max(cy - max_shift, 0), min(cy + max_shift + 1, corr.shape[0])
    x0, x1 = max(cx - max_shift, 0), min(cx + max_shift + 1, corr.shape[1])
    window = corr[y0:y1, x0:x1]
    py, px = np.unravel_index(np.argmax(window), window.shape)
    peak_y = y0 + py
    peak_x = x0 + px
    dy = float(peak_y - cy)
    dx = float(peak_x - cx)
    score = float(corr[peak_y, peak_x])
    return dy, dx, score


def subtract_detector_static(stack: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        static = nanmedian_no_warning(stack, axis=0)
    return stack - static[None]


def solve_global_position_corrections(n_frames: int, records: list[dict[str, float]]) -> np.ndarray:
    if not records:
        return np.zeros((n_frames, 2), dtype=np.float64)
    rows = []
    rhs_y = []
    rhs_x = []
    weights = []
    for rec in records:
        i = int(rec["i"])
        j = int(rec["j"])
        row = np.zeros(n_frames, dtype=np.float64)
        row[j] = 1.0
        row[i] = -1.0
        rows.append(row)
        rhs_y.append(rec["dy"])
        rhs_x.append(rec["dx"])
        weights.append(max(float(rec.get("score", 0.0)), 1e-6))

    # Anchor frame 0 to remove the arbitrary global translation gauge.
    anchor = np.zeros(n_frames, dtype=np.float64)
    anchor[0] = 1.0
    rows.append(anchor)
    rhs_y.append(0.0)
    rhs_x.append(0.0)
    weights.append(max(weights) * 10.0 if weights else 1.0)

    A = np.vstack(rows)
    w = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    Aw = A * w
    dy = np.linalg.lstsq(Aw, np.asarray(rhs_y, dtype=np.float64) * w[:, 0], rcond=None)[0]
    dx = np.linalg.lstsq(Aw, np.asarray(rhs_x, dtype=np.float64) * w[:, 0], rcond=None)[0]
    return np.stack([dy, dx], axis=1)


def refine_positions_pairwise(stack: np.ndarray, positions: np.ndarray, grid_y: int, grid_x: int, mode: str, max_shift: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    if mode == "none":
        return positions, []
    work = subtract_detector_static(stack) if mode == "static-removed" else stack
    records: list[dict[str, float]] = []

    def add_edge(i: int, j: int, axis: str) -> None:
        mask = np.isfinite(work[i]) & np.isfinite(work[j])
        dy, dx, score = phase_correlation_shift(work[i], work[j], mask, max_shift)
        records.append({"i": float(i), "j": float(j), "dy": dy, "dx": dx, "score": score, "axis": axis})

    for iy in range(grid_y):
        row_start = iy * grid_x
        for ix in range(1, grid_x):
            add_edge(row_start + ix - 1, row_start + ix, "x")
    for iy in range(1, grid_y):
        for ix in range(grid_x):
            add_edge((iy - 1) * grid_x + ix, iy * grid_x + ix, "y")

    corrections = solve_global_position_corrections(len(positions), records)
    for index, corr in enumerate(corrections):
        records.append({"i": float(index), "j": float(index), "dy": float(corr[0]), "dx": float(corr[1]), "score": 1.0, "axis": "global_correction"})
    return positions + corrections, records


def finite_to_float32(stack: np.ndarray) -> np.ndarray:
    return stack.astype(np.float32)


def save_tiff(path: Path, stack: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = finite_to_float32(stack)
    try:
        import tifffile
        tifffile.imwrite(path, arr, photometric="minisblack")
    except Exception:
        try:
            import imageio.v3 as iio
            iio.imwrite(path, arr)
        except Exception as exc:
            raise RuntimeError("Saving TIFF requires tifffile or imageio") from exc


def run(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    files = list_input_files(raw_dir, args.raw_glob)
    if args.raw_limit is not None:
        files = files[: args.raw_limit]
    if not files:
        raise FileNotFoundError(f"No files matched {args.raw_glob!r} in {raw_dir}")

    raw_stack, first_meta = load_raw_stack(files, args.cpu_cores)
    crop = parse_crop(args.crop)
    if crop is None and (args.gui_crop or args.manual_crop):
        preview = nanmedian_no_warning(raw_stack, axis=0)
        crop = manual_crop_gui(preview)
    if crop is None and args.auto_crop:
        crop = auto_crop_from_finite(raw_stack, args.auto_crop_margin, args.auto_crop_min_fraction)
    if crop is None:
        crop = (slice(None), slice(None))
    crop_box = crop_to_tuple(crop, raw_stack.shape[1:])

    crop_stack = raw_stack[:, crop[0], crop[1]]
    crop_masks = np.isfinite(crop_stack).astype(np.float64)

    calibration = load_calibration(args.calibration)
    calibration_raw = as_2d_calibration(calibration) if calibration is not None else None
    calibration_crop = None
    if calibration_raw is not None:
        if calibration_raw.shape != raw_stack.shape[1:]:
            raise ValueError(f"Calibration shape {calibration_raw.shape} does not match raw frame shape {raw_stack.shape[1:]}")
        calibration_crop = calibration_raw[crop[0], crop[1]]
        if args.raw_downsample > 1:
            calibration_crop = calibration_crop[:: args.raw_downsample, :: args.raw_downsample]

    meta = first_meta or {}
    pixel_spacing_x_value = args.raw_pixel_spacing_mm if args.raw_pixel_spacing_mm is not None else meta.get("xPixSpace")
    pixel_spacing_y_value = args.raw_pixel_spacing_y_mm if args.raw_pixel_spacing_y_mm is not None else meta.get("yPixSpace", pixel_spacing_x_value)
    if pixel_spacing_x_value is None or pixel_spacing_y_value is None:
        raise ValueError("Could not infer pixel spacing; pass --raw-pixel-spacing-mm and optionally --raw-pixel-spacing-y-mm")
    pixel_spacing_x_mm = float(pixel_spacing_x_value)
    pixel_spacing_y_mm = float(pixel_spacing_y_value)
    processed_pixel_spacing_x_mm = pixel_spacing_x_mm * args.raw_downsample
    processed_pixel_spacing_y_mm = pixel_spacing_y_mm * args.raw_downsample

    crop_stack_ds = downsample_stack(crop_stack, args.raw_downsample)
    crop_masks_ds = downsample_stack(crop_masks, args.raw_downsample)
    processed = crop_stack_ds.copy()
    if calibration_crop is not None:
        processed = processed - calibration_crop[None]
    processed_masks = np.isfinite(processed).astype(np.float64) * crop_masks_ds

    processed, low_order_coeffs = remove_low_order_stack(processed, processed_masks, args.remove_order, args.scan_axis, args.cpu_cores)
    processed, curvature_coeffs, curvature_trends, curvature_stats = remove_curvature_stack(processed, processed_masks, args.remove_curvature, args.cpu_cores, processed_pixel_spacing_y_mm, processed_pixel_spacing_x_mm)
    if args.subtract_static_for_output:
        processed = subtract_detector_static(processed)

    step_px = args.raw_step_mm / pixel_spacing_x_mm / args.raw_downsample
    nominal_positions = scan_positions_from_grid(len(files), args.raw_grid_y, args.raw_grid_x, step_px, args.raw_serpentine)
    positions, corr_records = refine_positions_pairwise(processed, nominal_positions, args.raw_grid_y, args.raw_grid_x, args.refine_positions, args.max_corr_shift)

    optic_shape = np.array([
        int(np.ceil(np.max(positions[:, 0]) + processed.shape[1] + 8)),
        int(np.ceil(np.max(positions[:, 1]) + processed.shape[2] + 8)),
    ])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tiff(output_dir / "raw_stack.tiff", raw_stack)
    save_tiff(output_dir / "crop_stack.tiff", crop_stack_ds)
    if calibration_raw is not None and calibration_crop is not None:
        save_tiff(output_dir / "calibration_raw.tiff", calibration_raw[None])
        save_tiff(output_dir / "calibration_crop.tiff", calibration_crop[None])
    save_tiff(output_dir / "processed_stack.tiff", processed)
    np.savez_compressed(
        output_dir / "processed_stitching_input.npz",
        measurements=processed,
        masks=processed_masks,
        scan_positions=positions,
        nominal_scan_positions=nominal_positions,
        optic_shape=optic_shape,
        raw_files=np.array([str(path) for path in files]),
        crop_box_y0_y1_x0_x1=np.array(crop_box),
        low_order_coeffs=low_order_coeffs,
        curvature_coeffs=curvature_coeffs,
        curvature_trends=curvature_trends,
        curvature_stats=np.array(json.dumps(curvature_stats, ensure_ascii=False)),
        correlation_records=np.array(corr_records, dtype=object),
        raw_pixel_spacing_mm=np.array(pixel_spacing_x_mm),
        raw_pixel_spacing_y_mm=np.array(pixel_spacing_y_mm),
        processed_pixel_spacing_x_mm=np.array(processed_pixel_spacing_x_mm),
        processed_pixel_spacing_y_mm=np.array(processed_pixel_spacing_y_mm),
        raw_step_mm=np.array(args.raw_step_mm),
        raw_downsample=np.array(args.raw_downsample),
        raw_grid_shape=np.array([args.raw_grid_y, args.raw_grid_x]),
        calibration_path=np.array(args.calibration or ""),
        calibration_raw=calibration_raw if calibration_raw is not None else np.array([]),
        calibration_crop=calibration_crop if calibration_crop is not None else np.array([]),
        preprocessing=json.dumps(vars(args), ensure_ascii=False),
    )
    summary = {
        "frames": len(files),
        "raw_shape": list(raw_stack.shape),
        "crop_shape": list(crop_stack.shape),
        "processed_shape": list(processed.shape),
        "crop_box_y0_y1_x0_x1": list(crop_box),
        "crop_string": f"{crop_box[0]}:{crop_box[1]}:{crop_box[2]}:{crop_box[3]}",
        "pixel_spacing_x_mm": pixel_spacing_x_mm,
        "pixel_spacing_y_mm": pixel_spacing_y_mm,
        "processed_pixel_spacing_x_mm": processed_pixel_spacing_x_mm,
        "processed_pixel_spacing_y_mm": processed_pixel_spacing_y_mm,
        "step_px_after_downsample": step_px,
        "remove_order": args.remove_order,
        "remove_curvature": args.remove_curvature,
        "curvature_stats": curvature_stats,
        "scan_axis": args.scan_axis,
        "calibration": args.calibration,
        "calibration_shape": list(calibration_raw.shape) if calibration_raw is not None else None,
        "calibration_crop_shape": list(calibration_crop.shape) if calibration_crop is not None else None,
        "calibration_raw_tiff": str(output_dir / "calibration_raw.tiff") if calibration_raw is not None else None,
        "calibration_crop_tiff": str(output_dir / "calibration_crop.tiff") if calibration_crop is not None else None,
        "refine_positions": args.refine_positions,
        "cpu_cores": args.cpu_cores,
        "output_npz": str(output_dir / "processed_stitching_input.npz"),
    }
    (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Curvature removal stats:")
    print(json.dumps(curvature_stats, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--raw-glob", default="*.asc")
    parser.add_argument("--output-dir", default="preprocessed_stitching")
    parser.add_argument("--raw-grid-y", type=int, default=3)
    parser.add_argument("--raw-grid-x", type=int, default=23)
    parser.add_argument("--raw-step-mm", type=float, default=4.0)
    parser.add_argument("--raw-pixel-spacing-mm", type=float, default=None, help="Raw X pixel spacing in mm. Defaults to ASC xPixSpace metadata.")
    parser.add_argument("--raw-pixel-spacing-y-mm", type=float, default=None, help="Raw Y pixel spacing in mm. Defaults to ASC yPixSpace metadata, then X spacing if absent.")
    parser.add_argument("--raw-downsample", type=int, default=4)
    parser.add_argument("--raw-limit", type=int, default=None)
    parser.add_argument("--raw-serpentine", action="store_true")
    parser.add_argument("--cpu-cores", type=int, default=16, help="CPU worker processes for ASC loading and independent per-frame detrending. Use 1 for serial debugging.")
    parser.add_argument("--crop", default=None, help="Crop as y0:y1:x0:x1 before downsampling.")
    parser.add_argument("--gui-crop", action="store_true", help="Open a matplotlib rectangle selector for crop selection.")
    parser.add_argument("--manual-crop", action="store_true", help="Deprecated alias for --gui-crop.")
    parser.add_argument("--auto-crop", action="store_true", help="Crop to finite-data bounding box.")
    parser.add_argument("--auto-crop-margin", type=int, default=8)
    parser.add_argument("--auto-crop-min-fraction", type=float, default=0.25)
    parser.add_argument("--calibration", default=None, help="ASC/NPZ/TIFF interferometer systematic to subtract before low-order removal.")
    parser.add_argument("--remove-order", default="tilt", choices=["none", "piston", "tilt"], help="2D low-order removal before curvature: piston is constant, tilt is a 2D plane.")
    parser.add_argument("--remove-curvature", default="none", choices=["none", "2d", "1dx", "1dy"], help="Pure curvature-only removal independent from piston/tilt: 2d=x^2,xy,y^2; 1dx=x^2; 1dy=y^2.")
    parser.add_argument("--scan-axis", default="x", choices=["x", "y"], help="Deprecated; retained for old command compatibility. Use --remove-curvature 1dx or 1dy.")
    parser.add_argument("--subtract-static-for-output", action="store_true", help="Subtract detector-coordinate median stack from final processed images.")
    parser.add_argument("--refine-positions", default="none", choices=["none", "global", "static-removed"], help="Pairwise phase-correlation position refinement. static-removed subtracts detector-fixed median before correlation.")
    parser.add_argument("--max-corr-shift", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
