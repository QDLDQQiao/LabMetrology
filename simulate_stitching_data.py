#!/usr/bin/env python3
"""Generate synthetic raster-scan interferometer stitching data.

The output is an ``.npz`` file consumed by ``stitching_interferometer_autodiff.py``.
It contains local subaperture measurements, masks, nominal and true scan
positions, and ground-truth optic/systematic/plane terms for verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_surface(shape: tuple[int, int], seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    surface = (
        1.8 * xx
        - 0.9 * yy
        + 0.7 * (2 * xx**2 + yy**2)
        + 0.45 * np.sin(2 * np.pi * (1.4 * xx + 0.25 * yy))
        + 0.25 * np.cos(2 * np.pi * (0.35 * xx - 1.7 * yy))
        + 0.12 * np.sin(2 * np.pi * (5.0 * xx + 2.3 * yy))
    )

    # Add smooth random texture by summing low-amplitude sinusoids.
    for _ in range(16):
        fx, fy = rng.uniform(0.5, 7.0, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.normal(scale=0.06)
        surface += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
    return scale * surface


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.floor(np.clip(x, 0, w - 1)).astype(int)
    y0 = np.floor(np.clip(y, 0, h - 1)).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(x - x0, 0, 1)
    wy = np.clip(y - y0, 0, 1)
    values = (
        (1 - wx) * (1 - wy) * image[y0, x0]
        + wx * (1 - wy) * image[y0, x1]
        + (1 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )
    values[~valid] = np.nan
    return values, valid


def local_mask(shape: tuple[int, int], radius: float = 0.48) -> np.ndarray:
    h, w = shape
    y = (np.arange(h) - (h - 1) / 2) / max(h, w)
    x = (np.arange(w) - (w - 1) / 2) / max(h, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return ((xx**2 + yy**2) <= radius**2).astype(float)


def generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    optic_shape = (args.canvas_h, args.canvas_w)
    frame_shape = (args.frame_h, args.frame_w)
    optic = make_surface(optic_shape, args.seed, scale=args.optic_scale)
    systematic = make_surface(frame_shape, args.seed + 11, scale=args.systematic_scale)
    mask = local_mask(frame_shape)

    positions = []
    if args.cover_full_aperture:
        # Ptychography-style extended object geometry: the probe/subaperture
        # centers span the full object support.  Top-left positions may be
        # negative, which is correct; samples outside the object are invalid,
        # while the circular aperture still constrains the object edge.
        center_y = np.linspace(0, args.canvas_h - 1, args.grid_y)
        center_x = np.linspace(0, args.canvas_w - 1, args.grid_x)
        for cy in center_y:
            for cx in center_x:
                positions.append([cy - (args.frame_h - 1) / 2, cx - (args.frame_w - 1) / 2])
    else:
        step_y = args.frame_h * (1 - args.overlap)
        step_x = args.frame_w * (1 - args.overlap)
        margin_y = args.margin
        margin_x = args.margin
        for iy in range(args.grid_y):
            for ix in range(args.grid_x):
                positions.append([margin_y + iy * step_y, margin_x + ix * step_x])
    nominal_positions = np.asarray(positions, dtype=float)
    true_positions = nominal_positions + rng.normal(scale=args.position_jitter, size=nominal_positions.shape)

    h, w = frame_shape
    yy, xx = np.indices(frame_shape, dtype=float)
    yy_norm = 2 * yy / (h - 1) - 1
    xx_norm = 2 * xx / (w - 1) - 1

    measurements = []
    masks = []
    planes = []
    for pos_y, pos_x in true_positions:
        optic_patch, inside = bilinear_sample(optic, yy + pos_y, xx + pos_x)
        piston = rng.normal(scale=args.plane_piston)
        tilt_x = rng.normal(scale=args.plane_tilt)
        tilt_y = rng.normal(scale=args.plane_tilt)
        plane = piston + tilt_x * xx_norm + tilt_y * yy_norm
        noise = rng.normal(scale=args.noise, size=frame_shape)
        measurement = optic_patch + systematic + plane + noise
        frame_mask = mask * inside.astype(float)
        measurement[frame_mask <= 0] = np.nan
        measurements.append(measurement)
        masks.append(frame_mask)
        planes.append([piston, tilt_x, tilt_y])

    measurements = np.asarray(measurements)
    masks = np.asarray(masks)
    planes = np.asarray(planes)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        measurements=measurements,
        masks=masks,
        scan_positions=nominal_positions,
        true_positions=true_positions,
        optic_true=optic,
        systematic_true=systematic,
        planes_true=planes,
        noise_sigma=args.noise,
        frame_shape=np.array(frame_shape),
        optic_shape=np.array(optic_shape),
        grid_shape=np.array([args.grid_y, args.grid_x]),
        overlap=args.overlap,
    )
    print(f"Wrote {output.resolve()}")
    print(f"measurements: {measurements.shape}; optic: {optic.shape}; systematic: {systematic.shape}")
    print(f"noise sigma: {args.noise}; position jitter sigma: {args.position_jitter} px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="stitching_simulation_data.npz")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--canvas-h", type=int, default=176)
    parser.add_argument("--canvas-w", type=int, default=288)
    parser.add_argument("--frame-h", type=int, default=64)
    parser.add_argument("--frame-w", type=int, default=64)
    parser.add_argument("--grid-y", type=int, default=3)
    parser.add_argument("--grid-x", type=int, default=5)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--margin", type=float, default=16.0)
    parser.add_argument(
        "--cover-full-aperture",
        action="store_true",
        help="Place subaperture centers across the full optic support, allowing negative top-left positions.",
    )
    parser.add_argument("--position-jitter", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--optic-scale", type=float, default=1.0)
    parser.add_argument("--systematic-scale", type=float, default=0.25)
    parser.add_argument("--plane-piston", type=float, default=0.3)
    parser.add_argument("--plane-tilt", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
