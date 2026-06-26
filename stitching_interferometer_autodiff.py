#!/usr/bin/env python3
"""PyTorch autodiff stitching reconstruction for raster-scan interferometry."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


@dataclass
class StitchingResult:
    optic: np.ndarray
    systematic: np.ndarray | None
    planes: np.ndarray
    positions: np.ndarray
    residuals: np.ndarray
    reconstructed: np.ndarray
    observed_mask: np.ndarray
    quality_mask: np.ndarray
    loss_history: list[dict[str, Any]]
    summary: dict[str, Any]


def finite_rms(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.sqrt(np.mean(values[valid] ** 2))) if np.any(valid) else float("nan")


def make_local_bases(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.ones(shape), xx, yy


def canvas_shape_from_positions(
    positions: np.ndarray,
    frame_shape: tuple[int, int],
    margin: int,
    requested: tuple[int, int] | None,
) -> tuple[int, int]:
    if requested is not None:
        return requested
    h, w = frame_shape
    max_y = float(np.max(positions[:, 0])) + h + margin
    max_x = float(np.max(positions[:, 1])) + w + margin
    return int(np.ceil(max_y)), int(np.ceil(max_x))


def warm_start_optic(
    measurements: np.ndarray,
    masks: np.ndarray,
    positions: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n, h, w = measurements.shape
    H, W = canvas_shape
    numerator = np.zeros((H, W), dtype=np.float64)
    denominator = np.zeros((H, W), dtype=np.float64)
    for index in range(n):
        py, px = np.round(positions[index]).astype(int)
        patch = measurements[index].copy()
        valid = np.isfinite(patch) & (masks[index] > 0)
        if np.any(valid):
            patch_mean = np.nanmean(patch[valid])
            patch = patch - patch_mean
        y0 = max(0, py)
        x0 = max(0, px)
        y1 = min(H, py + h)
        x1 = min(W, px + w)
        sy0 = y0 - py
        sx0 = x0 - px
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)
        local_valid = valid[sy0:sy1, sx0:sx1]
        local_weight = masks[index][sy0:sy1, sx0:sx1]
        numerator[y0:y1, x0:x1][local_valid] += patch[sy0:sy1, sx0:sx1][local_valid] * local_weight[local_valid]
        denominator[y0:y1, x0:x1][local_valid] += local_weight[local_valid]
    optic = np.zeros((H, W), dtype=np.float64)
    valid = denominator > 0
    optic[valid] = numerator[valid] / denominator[valid]
    return optic, denominator


def make_legendre2d_basis(
    shape: tuple[int, int],
    *,
    order: int = 10,
    order_x: int | None = None,
    order_y: int | None = None,
    total_degree: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Create 2D Legendre-product basis maps on normalized canvas coordinates."""

    H, W = shape
    ox = order if order_x is None else order_x
    oy = order if order_y is None else order_y
    x = np.linspace(-1.0, 1.0, W, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float64)
    vx = np.polynomial.legendre.legvander(x, ox)
    vy = np.polynomial.legendre.legvander(y, oy)
    maps = []
    terms = []
    for iy in range(oy + 1):
        for ix in range(ox + 1):
            if total_degree and order_x is None and order_y is None and ix + iy > order:
                continue
            maps.append(np.outer(vy[:, iy], vx[:, ix]))
            terms.append((ix, iy))
    return np.asarray(maps, dtype=np.float64), terms


def fit_basis_coefficients(
    image: np.ndarray,
    basis: np.ndarray,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares fit image ~= sum_k coeff[k] * basis[k]."""

    valid = np.isfinite(image)
    if weight is not None:
        valid &= np.isfinite(weight) & (weight > 0)
    if np.count_nonzero(valid) < basis.shape[0]:
        return np.zeros(basis.shape[0], dtype=np.float64)
    A = basis[:, valid].T
    b = image[valid]
    if weight is not None:
        w = np.sqrt(weight[valid])
        A = A * w[:, None]
        b = b * w
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coeff.astype(np.float64)


def remove_low_order_legendre(
    image: np.ndarray,
    *,
    order: int,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    basis, terms = make_legendre2d_basis(image.shape, order=order, total_degree=True)
    coeff = fit_basis_coefficients(image, basis, weight=weight)
    low = np.tensordot(coeff, basis, axes=(0, 0))
    high = image - low
    high[~np.isfinite(image)] = 0.0
    return high, coeff, terms


def scan_condition(positions: np.ndarray) -> tuple[float, np.ndarray]:
    centered = positions - positions.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[-1] < 1e-12:
        return float("inf"), np.array([1.0, 0.0])
    return float(singular[0] / singular[-1]), vh[-1]


def estimate_step_px(positions: np.ndarray) -> float:
    """Estimate the nominal scan step in pixels from nonzero scan increments."""

    deltas: list[np.ndarray] = []
    for axis in range(2):
        unique = np.unique(np.round(positions[:, axis].astype(np.float64), decimals=6))
        diff = np.diff(np.sort(unique))
        diff = diff[diff > 1e-6]
        if diff.size:
            deltas.append(diff)
    if not deltas:
        return 4.0
    return float(np.median(np.concatenate(deltas)))


def step_px_from_data(data: dict[str, Any]) -> float | None:
    """Read nominal scan step from preprocessor metadata when available."""

    required = ("raw_step_mm", "raw_pixel_spacing_mm", "raw_downsample")
    if not all(key in data for key in required):
        return None
    try:
        raw_step_mm = float(np.asarray(data["raw_step_mm"]).reshape(-1)[0])
        pixel_spacing_mm = float(np.asarray(data["raw_pixel_spacing_mm"]).reshape(-1)[0])
        downsample = float(np.asarray(data["raw_downsample"]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    if pixel_spacing_mm <= 0 or downsample <= 0:
        return None
    return raw_step_mm / pixel_spacing_mm / downsample


def erode_binary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return eroded


def apodize_masks(masks: np.ndarray, radius: int) -> np.ndarray:
    """Cosine-feather binary apertures to suppress stitching edge imprint."""

    if radius <= 0:
        return masks.astype(np.float64)
    output = np.zeros_like(masks, dtype=np.float64)
    for index, mask in enumerate(masks > 0):
        distance = np.zeros(mask.shape, dtype=np.float64)
        current = mask.copy()
        for step in range(radius):
            eroded = erode_binary(current)
            ring = current & ~eroded
            distance[ring] = step
            current = eroded
        distance[current] = radius
        t = np.clip(distance / radius, 0.0, 1.0)
        weight = 0.5 - 0.5 * np.cos(np.pi * t)
        weight[~mask] = 0.0
        output[index] = weight
    return output


class StitchingModel(nn.Module):
    def __init__(
        self,
        measurements: np.ndarray,
        masks: np.ndarray,
        scan_positions: np.ndarray,
        optic_canvas_shape: tuple[int, int],
        *,
        S_known: np.ndarray | None = None,
        fit_S: bool = False,
        fit_positions: bool = False,
        fit_power_per_frame: bool = False,
        plane_order: str = "tilt",
        interp_mode: str = "bilinear",
        aperture_feather: int = 4,
        min_output_coverage: float = 0.15,
        pixel_spacing: tuple[float, float] | None = None,
        known_radius_m: float | None = None,
        known_radius_axis: str = "x",
        step_px: float | None = None,
        initial_optic: np.ndarray | None = None,
        anchor_curvature: bool = True,
        optic_model: str = "pixel",
        optic_poly_order: int = 10,
        optic_poly_order_x: int | None = None,
        optic_poly_order_y: int | None = None,
        optic_poly_basis: str = "legendre",
        device: str = "cpu",
    ):
        super().__init__()
        device_t = torch.device(device)
        finite = np.isfinite(measurements)
        clean_M = np.where(finite, measurements, 0.0)
        binary_mask = np.where(finite, masks, 0.0)
        clean_mask = binary_mask.astype(np.float64, copy=False)

        self.N, self.h, self.w = clean_M.shape
        self.H, self.W = optic_canvas_shape
        self.fit_S = fit_S
        self.fit_positions = fit_positions
        self.fit_power_per_frame = fit_power_per_frame
        self.plane_order = plane_order
        self.fit_tilt_per_frame = plane_order == "tilt" or fit_power_per_frame
        self.interp_mode = interp_mode
        self.optic_model = optic_model
        self.optic_poly_order = optic_poly_order
        self.optic_poly_order_x = optic_poly_order_x
        self.optic_poly_order_y = optic_poly_order_y
        self.optic_poly_basis = optic_poly_basis
        self.O_terms: list[tuple[int, int]] | None = None
        if optic_model not in ("pixel", "poly"):
            raise ValueError(f"unsupported optic_model {optic_model!r}")
        if optic_poly_basis != "legendre":
            raise ValueError("only --optic-poly-basis legendre is currently implemented")
        self.scan_is_1d = float(np.ptp(scan_positions[:, 0])) < 0.5
        nominal_step_px = estimate_step_px(scan_positions) if step_px is None else float(step_px)
        self.max_delta_pos = float(max(4.0, 2.0 * nominal_step_px))

        self.register_buffer("M", torch.as_tensor(clean_M, dtype=torch.float64, device=device_t))
        self.register_buffer("mask", torch.as_tensor(clean_mask, dtype=torch.float64, device=device_t))
        self.register_buffer("scan_positions", torch.as_tensor(scan_positions, dtype=torch.float64, device=device_t))

        _, tx, ty = make_local_bases((self.h, self.w))
        rr = tx**2 + ty**2
        self.register_buffer("tilt_u", torch.as_tensor(tx, dtype=torch.float64, device=device_t))
        self.register_buffer("tilt_v", torch.as_tensor(ty, dtype=torch.float64, device=device_t))
        self.register_buffer("power_basis", torch.as_tensor(rr - np.mean(rr), dtype=torch.float64, device=device_t))
        s_bases = np.stack([
            np.ones_like(tx),
            tx,
            ty,
            tx**2 - np.mean(tx**2),
            ty**2 - np.mean(ty**2),
            tx * ty,
        ], axis=0)
        q, _ = np.linalg.qr(s_bases.reshape(6, -1).T)
        self.register_buffer("S_low_bases", torch.as_tensor(q.T.reshape(6, self.h, self.w), dtype=torch.float64, device=device_t))

        if S_known is not None:
            self.register_buffer("S_fixed", torch.as_tensor(S_known, dtype=torch.float64, device=device_t))
        else:
            self.S_fixed = None

        warm, soft_coverage = warm_start_optic(clean_M, clean_mask, scan_positions, optic_canvas_shape)
        _, binary_coverage = warm_start_optic(clean_M, binary_mask, scan_positions, optic_canvas_shape)
        optic_prior = None
        optic_prior_mask = None
        if initial_optic is not None:
            if initial_optic.shape != warm.shape:
                raise ValueError(f"initial_optic shape {initial_optic.shape} does not match canvas {warm.shape}")
            init = np.asarray(initial_optic, dtype=np.float64).copy()
            prior_valid = np.isfinite(init) & (binary_coverage > 0)
            init[~np.isfinite(init)] = warm[~np.isfinite(init)]
            warm = init
            optic_prior = np.where(prior_valid, init, 0.0)
            optic_prior_mask = prior_valid.astype(np.float64)
        self.O: nn.Parameter | None = None
        self.O_coeff: nn.Parameter | None = None
        if optic_prior is not None and optic_prior_mask is not None:
            self.register_buffer("optic_prior", torch.as_tensor(optic_prior, dtype=torch.float64, device=device_t))
            self.register_buffer("optic_prior_mask", torch.as_tensor(optic_prior_mask, dtype=torch.float64, device=device_t))
        else:
            self.optic_prior = None
            self.optic_prior_mask = None
        binary_coverage_norm = binary_coverage / max(float(np.max(binary_coverage)), 1.0)
        soft_coverage_norm = soft_coverage / max(float(np.max(soft_coverage)), 1.0)
        self.min_output_coverage = min_output_coverage
        self.register_buffer("optic_observed_mask", torch.as_tensor(binary_coverage_norm, dtype=torch.float64, device=device_t))
        self.register_buffer("optic_quality_mask", torch.as_tensor(soft_coverage_norm, dtype=torch.float64, device=device_t))
        coverage_t = torch.as_tensor(binary_coverage_norm, dtype=torch.float64, device=device_t)
        obs_t = (coverage_t > 0).to(torch.float64)
        grad_valid_x = obs_t[:, 1:] * obs_t[:, :-1]
        grad_valid_y = obs_t[1:, :] * obs_t[:-1, :]
        seam_weight_x = grad_valid_x * torch.abs(coverage_t[:, 1:] - coverage_t[:, :-1])
        seam_weight_y = grad_valid_y * torch.abs(coverage_t[1:, :] - coverage_t[:-1, :])
        seam_count_x = (seam_weight_x > 0).to(torch.float64).sum()
        seam_count_y = (seam_weight_y > 0).to(torch.float64).sum()
        seam_mean_x = seam_weight_x.sum() / seam_count_x.clamp_min(1.0)
        seam_mean_y = seam_weight_y.sum() / seam_count_y.clamp_min(1.0)
        seam_weight_x = torch.where(seam_count_x > 0, seam_weight_x / seam_mean_x.clamp_min(1e-12), seam_weight_x)
        seam_weight_y = torch.where(seam_count_y > 0, seam_weight_y / seam_mean_y.clamp_min(1e-12), seam_weight_y)
        lap_valid_x = obs_t[:, 2:] * obs_t[:, 1:-1] * obs_t[:, :-2]
        lap_valid_y = obs_t[2:, :] * obs_t[1:-1, :] * obs_t[:-2, :]
        seam_lap_weight_x = lap_valid_x * torch.maximum(
            torch.abs(coverage_t[:, 2:] - coverage_t[:, 1:-1]),
            torch.abs(coverage_t[:, 1:-1] - coverage_t[:, :-2]),
        )
        seam_lap_weight_y = lap_valid_y * torch.maximum(
            torch.abs(coverage_t[2:, :] - coverage_t[1:-1, :]),
            torch.abs(coverage_t[1:-1, :] - coverage_t[:-2, :]),
        )
        seam_lap_count_x = (seam_lap_weight_x > 0).to(torch.float64).sum()
        seam_lap_count_y = (seam_lap_weight_y > 0).to(torch.float64).sum()
        seam_lap_mean_x = seam_lap_weight_x.sum() / seam_lap_count_x.clamp_min(1.0)
        seam_lap_mean_y = seam_lap_weight_y.sum() / seam_lap_count_y.clamp_min(1.0)
        seam_lap_weight_x = torch.where(seam_lap_count_x > 0, seam_lap_weight_x / seam_lap_mean_x.clamp_min(1e-12), seam_lap_weight_x)
        seam_lap_weight_y = torch.where(seam_lap_count_y > 0, seam_lap_weight_y / seam_lap_mean_y.clamp_min(1e-12), seam_lap_weight_y)
        self.register_buffer("optic_grad_valid_x", grad_valid_x)
        self.register_buffer("optic_grad_valid_y", grad_valid_y)
        self.register_buffer("optic_seam_weight_x", seam_weight_x)
        self.register_buffer("optic_seam_weight_y", seam_weight_y)
        self.register_buffer("optic_lap_valid_x", lap_valid_x)
        self.register_buffer("optic_lap_valid_y", lap_valid_y)
        self.register_buffer("optic_seam_lap_weight_x", seam_lap_weight_x)
        self.register_buffer("optic_seam_lap_weight_y", seam_lap_weight_y)

        if optic_model == "pixel":
            self.O = nn.Parameter(torch.as_tensor(warm, dtype=torch.float64, device=device_t))
        else:
            total_degree = optic_poly_order_x is None and optic_poly_order_y is None
            basis_np, terms = make_legendre2d_basis(
                optic_canvas_shape,
                order=optic_poly_order,
                order_x=optic_poly_order_x,
                order_y=optic_poly_order_y,
                total_degree=total_degree,
            )
            coeff_np = fit_basis_coefficients(warm, basis_np, weight=binary_coverage_norm)
            self.register_buffer("O_basis", torch.as_tensor(basis_np, dtype=torch.float64, device=device_t))
            self.O_terms = terms
            self.O_coeff = nn.Parameter(torch.as_tensor(coeff_np, dtype=torch.float64, device=device_t))

        self.S = nn.Parameter(torch.zeros((self.h, self.w), dtype=torch.float64, device=device_t)) if fit_S else None
        k = 1 + (2 if self.fit_tilt_per_frame else 0) + (1 if fit_power_per_frame else 0)
        self.power_col = k - 1 if fit_power_per_frame else None
        self.planes = nn.Parameter(torch.zeros((self.N, k), dtype=torch.float64, device=device_t))
        self.delta_pos = nn.Parameter(torch.zeros((self.N, 2), dtype=torch.float64, device=device_t)) if fit_positions else None
        if fit_positions:
            scan_x = scan_positions[:, 1].astype(np.float64)
            if np.ptp(scan_x) > 0:
                scan_x_norm = 2.0 * (scan_x - scan_x.min()) / np.ptp(scan_x) - 1.0
            else:
                scan_x_norm = np.zeros_like(scan_x)
            position_design = np.column_stack([np.ones(self.N, dtype=np.float64), scan_x_norm])
            self.register_buffer("position_gauge_design", torch.as_tensor(position_design, dtype=torch.float64, device=device_t))
        else:
            self.position_gauge_design = None
        self.curvature_anchor_enabled = bool(optic_model == "pixel" and anchor_curvature and initial_optic is not None)
        self._setup_curvature_anchor()
        self._setup_known_radius_constraint(pixel_spacing, known_radius_m, known_radius_axis)
        if initial_optic is not None:
            self.initialize_planes_from_current_optic()

    @property
    def S_effective(self) -> torch.Tensor | None:
        if not self.fit_S or self.S is None:
            return None
        proj = (self.S[None] * self.S_low_bases).sum(dim=(1, 2))
        return self.S - (proj[:, None, None] * self.S_low_bases).sum(dim=0)

    def current_optic_map(self) -> torch.Tensor:
        if self.optic_model == "pixel":
            if self.O is None:
                raise RuntimeError("pixel optic model has no O parameter")
            return self.O
        if self.O_coeff is None:
            raise RuntimeError("polynomial optic model has no coefficient parameter")
        optic = torch.einsum("k,khw->hw", self.O_coeff, self.O_basis)
        return optic

    def optic_parameters(self) -> list[nn.Parameter]:
        if self.optic_model == "pixel":
            return [self.O] if self.O is not None else []
        params = [self.O_coeff] if self.O_coeff is not None else []
        return params

    def _setup_curvature_anchor(self) -> None:
        device = self.M.device
        y = torch.linspace(-1.0, 1.0, self.H, dtype=torch.float64, device=device)
        x = torch.linspace(-1.0, 1.0, self.W, dtype=torch.float64, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("optic_poly_x2", xx**2)
        self.register_buffer("optic_poly_y2", yy**2)
        valid = self.optic_observed_mask > 0
        design = torch.stack([
            torch.ones_like(xx)[valid],
            xx[valid],
            yy[valid],
            (xx**2)[valid],
            (yy**2)[valid],
        ], dim=1)
        self.register_buffer("optic_poly_valid", valid)
        self.register_buffer("optic_poly_design", design)
        affine_design = design[:, :3]
        affine_gram = affine_design.T @ affine_design
        affine_gram = affine_gram + 1e-12 * torch.eye(3, dtype=torch.float64, device=device)
        self.register_buffer("optic_affine_design", affine_design)
        self.register_buffer("optic_affine_gram_inv", torch.linalg.inv(affine_gram))
        with torch.no_grad():
            self.register_buffer("curvature_prior_coeff", self._optic_poly_coeff(self.current_optic_map()).detach())

    def _optic_poly_coeff(self, image: torch.Tensor) -> torch.Tensor:
        values = image[self.optic_poly_valid].reshape(-1, 1)
        if values.numel() < 5:
            return torch.zeros(5, dtype=torch.float64, device=image.device)
        return torch.linalg.lstsq(self.optic_poly_design, values).solution[:, 0]

    def _setup_known_radius_constraint(
        self,
        pixel_spacing: tuple[float, float] | None,
        known_radius_m: float | None,
        known_radius_axis: str,
    ) -> None:
        self.known_radius_enabled = False
        self.known_radius_m = None
        self.known_radius_axis = known_radius_axis
        self.known_radius_valid = None
        self.known_radius_pinv = None
        self.known_radius_target_q = None
        if known_radius_m is None:
            return
        if known_radius_axis not in ("x", "y", "principal"):
            raise ValueError("known_radius_axis must be x, y, or principal")
        radius_m = float(known_radius_m)
        if not np.isfinite(radius_m) or abs(radius_m) < 1e-30:
            raise ValueError("known_radius_m must be finite and nonzero")
        sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
        if not np.isfinite(sy_mm) or not np.isfinite(sx_mm) or sy_mm <= 0 or sx_mm <= 0:
            raise ValueError("known-radius constraint requires positive finite pixel spacing in mm")
        device = self.M.device
        y = (torch.arange(self.H, dtype=torch.float64, device=device) - (self.H - 1) / 2.0) * float(sy_mm)
        x = (torch.arange(self.W, dtype=torch.float64, device=device) - (self.W - 1) / 2.0) * float(sx_mm)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        valid = self.optic_observed_mask > 0
        if int(valid.sum().item()) < 6:
            return
        design = torch.stack([
            torch.ones_like(xx)[valid],
            xx[valid],
            yy[valid],
            (xx**2)[valid],
            (xx * yy)[valid],
            (yy**2)[valid],
        ], dim=1)
        self.known_radius_valid = valid
        self.known_radius_pinv = torch.linalg.pinv(design)
        self.known_radius_target_q = torch.as_tensor(500.0 / radius_m, dtype=torch.float64, device=device)
        self.known_radius_enabled = True
        self.known_radius_m = radius_m
        self.known_radius_axis = known_radius_axis

    def known_radius_coeff(self, image: torch.Tensor) -> torch.Tensor:
        if not self.known_radius_enabled or self.known_radius_valid is None or self.known_radius_pinv is None:
            return torch.zeros(6, dtype=torch.float64, device=image.device)
        values = image[self.known_radius_valid]
        return self.known_radius_pinv @ values

    def known_radius_current_q(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.known_radius_coeff(image)
        if self.known_radius_axis == "x":
            return coeff[3], coeff
        if self.known_radius_axis == "y":
            return coeff[5], coeff
        quad = torch.stack([
            torch.stack([coeff[3], 0.5 * coeff[4]]),
            torch.stack([0.5 * coeff[4], coeff[5]]),
        ])
        eigvals = torch.linalg.eigvalsh(quad)
        if self.known_radius_m is not None and self.known_radius_m < 0.0:
            return eigvals[0], coeff
        return eigvals[-1], coeff

    def project_curvature_to_prior(self) -> None:
        if not self.curvature_anchor_enabled or self.O is None:
            return
        with torch.no_grad():
            coeff = self._optic_poly_coeff(self.current_optic_map())
            dqxx = self.curvature_prior_coeff[3] - coeff[3]
            dqyy = self.curvature_prior_coeff[4] - coeff[4]
            self.O += dqxx * self.optic_poly_x2 + dqyy * self.optic_poly_y2

    def _plane_from_coeffs(self, pl: torch.Tensor) -> torch.Tensor:
        plane = pl[:, 0, None, None]
        if self.fit_tilt_per_frame:
            plane = plane + pl[:, 1, None, None] * self.tilt_u[None] + pl[:, 2, None, None] * self.tilt_v[None]
        if self.fit_power_per_frame and self.power_col is not None:
            plane = plane + pl[:, self.power_col, None, None] * self.power_basis[None]
        return plane

    def _sample_optic(self, frame_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid, inside = self.build_grid(frame_idx)
        optic_map = self.current_optic_map()
        optic_in = optic_map[None, None].expand(len(frame_idx), -1, -1, -1)
        optic_sub = F.grid_sample(
            optic_in,
            grid,
            mode=self.interp_mode,
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        return optic_sub, inside

    def _current_systematic_term(self) -> torch.Tensor | float:
        if self.fit_S:
            s_eff = self.S_effective
            return s_eff if s_eff is not None else 0.0
        if self.S_fixed is not None:
            return self.S_fixed
        return 0.0

    def initialize_planes_from_current_optic(self) -> None:
        with torch.no_grad():
            s_term = self._current_systematic_term()
            for frame in range(self.N):
                idx = torch.tensor([frame], dtype=torch.long, device=self.M.device)
                optic_sub, inside = self._sample_optic(idx)
                optic_sub = optic_sub.squeeze(0)
                valid = (self.mask[frame] * inside[0]) > 0
                if int(valid.sum().item()) < self.planes.shape[1]:
                    continue
                residual = self.M[frame] - optic_sub - s_term
                cols = [torch.ones_like(self.tilt_u)[valid]]
                if self.fit_tilt_per_frame:
                    cols += [self.tilt_u[valid], self.tilt_v[valid]]
                if self.fit_power_per_frame:
                    cols.append(self.power_basis[valid])
                design = torch.stack(cols, dim=1)
                coeff = torch.linalg.lstsq(design, residual[valid].reshape(-1, 1)).solution[:, 0]
                self.planes[frame, : coeff.numel()] = coeff

    def refresh_optic_priors(self) -> None:
        with torch.no_grad():
            optic_map = self.current_optic_map().detach()
            if self.optic_prior is not None and self.optic_prior_mask is not None:
                self.optic_prior.copy_(torch.where(self.optic_prior_mask > 0, optic_map, self.optic_prior))
            if self.curvature_anchor_enabled:
                self.curvature_prior_coeff.copy_(self._optic_poly_coeff(optic_map).detach())

    def project_position_gauge(self) -> None:
        """Remove unobservable position gauges from the refinement parameters.

        On a 1-D scan, an affine trend in x-position corrections changes the
        effective scan origin/step and can trade directly against global
        curvature. Keep only residual frame-to-frame corrections.
        """

        if self.delta_pos is None:
            return
        with torch.no_grad():
            if self.scan_is_1d:
                self.delta_pos[:, 0].zero_()
                if self.N >= 2 and self.position_gauge_design is not None:
                    design = self.position_gauge_design
                    coeff = torch.linalg.lstsq(design, self.delta_pos[:, 1:2]).solution
                    self.delta_pos[:, 1] -= (design @ coeff)[:, 0]
            else:
                self.delta_pos -= self.delta_pos.mean(dim=0, keepdim=True)

    def build_grid(self, frame_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(frame_idx)
        pos = self.scan_positions[frame_idx]
        if self.fit_positions:
            # Soft clip to avoid frames walking out of the canvas.
            pos = pos + self.max_delta_pos * torch.tanh(self.delta_pos[frame_idx] / self.max_delta_pos)
        y = torch.arange(self.h, dtype=torch.float64, device=pos.device)
        x = torch.arange(self.w, dtype=torch.float64, device=pos.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        canvas_y = yy[None] + pos[:, 0, None, None]
        canvas_x = xx[None] + pos[:, 1, None, None]
        grid_x = 2.0 * canvas_x / (self.W - 1) - 1.0
        grid_y = 2.0 * canvas_y / (self.H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        eps_x = 1.0 / (self.W - 1)
        eps_y = 1.0 / (self.H - 1)
        factor = 2.0 if self.interp_mode == "bicubic" else 1.0
        inside = ((grid_x.abs() <= 1.0 - factor * eps_x) & (grid_y.abs() <= 1.0 - factor * eps_y)).to(torch.float64)
        return grid, inside

    def forward_batch(self, frame_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        optic_sub, inside = self._sample_optic(frame_idx)
        plane = self._plane_from_coeffs(self.planes[frame_idx])

        if self.fit_S:
            s_eff = self.S_effective
            s_term = s_eff[None] if s_eff is not None else 0.0
        elif self.S_fixed is not None:
            s_term = self.S_fixed[None]
        else:
            s_term = 0.0

        prediction = optic_sub + s_term + plane
        total_mask = self.mask[frame_idx] * inside
        return prediction, total_mask, frame_idx

    def current_positions(self) -> torch.Tensor:
        if not self.fit_positions:
            return self.scan_positions
        return self.scan_positions + self.max_delta_pos * torch.tanh(self.delta_pos / self.max_delta_pos)

    def project_mean_plane_to_optic(self) -> None:
        """Move mean local piston/tilt gauge from frame planes into O.

        A canvas affine surface sampled by every frame appears as the same
        local tilt in each subaperture, plus a position-dependent piston.
        Without this projection, the optimizer can leave real low-order optic
        content in the per-frame planes, making the recovered optic amplitude
        too small while preserving a good residual.
        """

        if self.optic_model != "pixel" or self.O is None:
            return
        with torch.no_grad():
            mean_piston = self.planes[:, 0].mean()
            mean_tilt_u = self.planes[:, 1].mean() if self.fit_tilt_per_frame else torch.zeros((), dtype=torch.float64, device=self.M.device)
            mean_tilt_v = self.planes[:, 2].mean() if self.fit_tilt_per_frame else torch.zeros((), dtype=torch.float64, device=self.M.device)
            mean_power = self.planes[:, self.power_col].mean() if self.fit_power_per_frame and self.power_col is not None else None

            y = torch.arange(self.H, dtype=torch.float64, device=self.M.device)
            x = torch.arange(self.W, dtype=torch.float64, device=self.M.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            x_norm = 2.0 * xx / (self.W - 1) - 1.0
            y_norm = 2.0 * yy / (self.H - 1) - 1.0

            local_to_canvas_x = (self.w - 1) / (self.W - 1)
            local_to_canvas_y = (self.h - 1) / (self.H - 1)
            slope_x = mean_tilt_u / local_to_canvas_x
            slope_y = mean_tilt_v / local_to_canvas_y

            self.O += mean_piston + slope_x * x_norm + slope_y * y_norm

            pos = self.current_positions()
            center_x_norm = 2.0 * (pos[:, 1] + (self.w - 1) / 2.0) / (self.W - 1) - 1.0
            center_y_norm = 2.0 * (pos[:, 0] + (self.h - 1) / 2.0) / (self.H - 1) - 1.0
            induced_piston = mean_piston + slope_x * center_x_norm + slope_y * center_y_norm
            induced_tilt_u = torch.full_like(induced_piston, mean_tilt_u)
            induced_tilt_v = torch.full_like(induced_piston, mean_tilt_v)

            if self.fit_power_per_frame and mean_power is not None:
                quad_x = mean_power / (local_to_canvas_x**2)
                quad_y = mean_power / (local_to_canvas_y**2)
                self.O += quad_x * x_norm**2 + quad_y * y_norm**2

                local_rr_mean = ((self.tilt_u**2 + self.tilt_v**2) - self.power_basis).mean()
                induced_piston = (
                    induced_piston
                    + quad_x * center_x_norm**2
                    + quad_y * center_y_norm**2
                    + mean_power * local_rr_mean
                )
                induced_tilt_u = induced_tilt_u + 2.0 * quad_x * center_x_norm * local_to_canvas_x
                induced_tilt_v = induced_tilt_v + 2.0 * quad_y * center_y_norm * local_to_canvas_y
                self.planes[:, self.power_col] -= mean_power

            self.planes[:, 0] -= induced_piston
            if self.fit_tilt_per_frame:
                self.planes[:, 1] -= induced_tilt_u
                self.planes[:, 2] -= induced_tilt_v


def batched_loss(
    model: StitchingModel,
    frame_batch_size: int,
    lam_g: float,
    lam_s: float,
    smoothness: float,
    seam_smoothness: float,
    plane_l2: float,
    s_l2: float,
    s_smoothness: float,
    optic_prior_l2: float,
    position_l2: float,
    curv_anchor_l2: float,
    known_radius_l2: float,
    optic_coeff_l2: float,
    optic_coeff_degree_power: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    device = model.M.device
    optic_map = model.current_optic_map()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_sum = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, model.N, frame_batch_size):
        idx = torch.arange(start, min(model.N, start + frame_batch_size), device=device)
        prediction, mask, frame_idx = model.forward_batch(idx)
        residual = (model.M[frame_idx] - prediction) * mask
        loss_sum = loss_sum + (residual**2).sum()
        valid_sum = valid_sum + mask.sum()
    loss_data = loss_sum / valid_sum.clamp_min(1.0)

    observed = model.optic_observed_mask
    loss_univ_gauge = (
        ((optic_map * observed).sum() / observed.sum().clamp_min(1.0)) ** 2
        + model.planes[:, 0].mean() ** 2
    )
    if model.fit_tilt_per_frame:
        loss_univ_gauge = loss_univ_gauge + model.planes[:, 1].mean() ** 2 + model.planes[:, 2].mean() ** 2
    if model.fit_power_per_frame and model.power_col is not None:
        loss_univ_gauge = loss_univ_gauge + model.planes[:, model.power_col].mean() ** 2

    if model.scan_is_1d:
        y = torch.linspace(-1.0, 1.0, optic_map.shape[0], dtype=torch.float64, device=device)
        y_map = y[:, None]
        cov = observed
        cy = (optic_map * cov * y_map).sum() / (cov * y_map**2).sum().clamp_min(1.0)
        cy2 = (optic_map * cov * y_map**2).sum() / (cov * y_map**4).sum().clamp_min(1.0)
        loss_univ_gauge = loss_univ_gauge + 1e3 * (cy**2 + cy2**2)

    loss_S_gauge = torch.zeros((), dtype=torch.float64, device=device)
    if model.fit_S and model.S is not None:
        proj = (model.S_effective[None] * model.S_low_bases).sum(dim=(1, 2))
        loss_S_gauge = (proj**2).sum()

    if model.planes.shape[1] > 1:
        raw_plane_l2 = (model.planes[:, 1:]**2).mean()
    else:
        raw_plane_l2 = torch.zeros((), dtype=torch.float64, device=device)
    loss_plane_l2 = plane_l2 * raw_plane_l2

    loss_optic_prior = torch.zeros((), dtype=torch.float64, device=device)
    if optic_prior_l2 > 0 and model.optic_prior is not None and model.optic_prior_mask is not None:
        prior_valid = model.optic_prior_mask[model.optic_poly_valid] > 0
        if bool(prior_valid.any()):
            prior_values = (optic_map - model.optic_prior)[model.optic_poly_valid][prior_valid]
            design_affine = model.optic_affine_design[prior_valid]
            if prior_values.numel() >= 3:
                gram = design_affine.T @ design_affine
                gram = gram + 1e-12 * torch.eye(3, dtype=torch.float64, device=device)
                coeff_affine = torch.linalg.solve(gram, design_affine.T @ prior_values)
                prior_values = prior_values - design_affine @ coeff_affine
            loss_optic_prior = optic_prior_l2 * (prior_values**2).mean()

    loss_position_prior = torch.zeros((), dtype=torch.float64, device=device)
    if position_l2 > 0 and model.delta_pos is not None:
        delta = model.current_positions() - model.scan_positions
        loss_position_prior = position_l2 * (delta**2).mean()

    raw_smooth = torch.zeros((), dtype=torch.float64, device=device)
    if smoothness > 0:
        d2x = optic_map[:, 2:] - 2.0 * optic_map[:, 1:-1] + optic_map[:, :-2]
        d2y = optic_map[2:, :] - 2.0 * optic_map[1:-1, :] + optic_map[:-2, :]
        raw_smooth = (d2x**2 * model.optic_lap_valid_x).sum() / model.optic_lap_valid_x.sum().clamp_min(1.0)
        raw_smooth = raw_smooth + (d2y**2 * model.optic_lap_valid_y).sum() / model.optic_lap_valid_y.sum().clamp_min(1.0)
    loss_smooth = smoothness * raw_smooth

    raw_seam_smooth = torch.zeros((), dtype=torch.float64, device=device)
    if seam_smoothness > 0:
        d2x = optic_map[:, 2:] - 2.0 * optic_map[:, 1:-1] + optic_map[:, :-2]
        d2y = optic_map[2:, :] - 2.0 * optic_map[1:-1, :] + optic_map[:-2, :]
        raw_seam_smooth = (d2x**2 * model.optic_seam_lap_weight_x).sum() / (model.optic_seam_lap_weight_x > 0).to(torch.float64).sum().clamp_min(1.0)
        raw_seam_smooth = raw_seam_smooth + (d2y**2 * model.optic_seam_lap_weight_y).sum() / (model.optic_seam_lap_weight_y > 0).to(torch.float64).sum().clamp_min(1.0)
    loss_seam_smooth = seam_smoothness * raw_seam_smooth

    loss_s_prior = torch.zeros((), dtype=torch.float64, device=device)
    if model.fit_S and model.S is not None:
        s_eff = model.S_effective
        if s_eff is not None:
            if s_l2 > 0:
                loss_s_prior = loss_s_prior + s_l2 * (s_eff**2).mean()
            if s_smoothness > 0:
                loss_s_prior = loss_s_prior + s_smoothness * (
                    ((s_eff[:, 1:] - s_eff[:, :-1]) ** 2).mean()
                    + ((s_eff[1:] - s_eff[:-1]) ** 2).mean()
                )

    loss_curv_anchor = torch.zeros((), dtype=torch.float64, device=device)
    if curv_anchor_l2 > 0 and model.curvature_anchor_enabled:
        coeff = model._optic_poly_coeff(optic_map)
        dq = coeff[3:5] - model.curvature_prior_coeff[3:5]
        loss_curv_anchor = curv_anchor_l2 * (dq**2).sum()

    loss_known_radius = torch.zeros((), dtype=torch.float64, device=device)
    raw_known_radius = torch.zeros((), dtype=torch.float64, device=device)
    if known_radius_l2 > 0 and model.known_radius_enabled and model.known_radius_target_q is not None:
        current_q, _ = model.known_radius_current_q(optic_map)
        raw_known_radius = (current_q - model.known_radius_target_q) ** 2
        loss_known_radius = known_radius_l2 * raw_known_radius

    loss_optic_coeff = torch.zeros((), dtype=torch.float64, device=device)
    if model.optic_model != "pixel" and optic_coeff_l2 > 0 and model.O_coeff is not None and model.O_terms is not None:
        degrees = torch.as_tensor([ix + iy for ix, iy in model.O_terms], dtype=torch.float64, device=device)
        weights = (1.0 + degrees) ** optic_coeff_degree_power
        loss_optic_coeff = optic_coeff_l2 * (weights * model.O_coeff**2).mean()

    weighted_univ_gauge = lam_g * loss_univ_gauge
    weighted_S_gauge = lam_s * loss_S_gauge
    total = (
        loss_data
        + weighted_univ_gauge
        + weighted_S_gauge
        + loss_plane_l2
        + loss_optic_prior
        + loss_position_prior
        + loss_smooth
        + loss_seam_smooth
        + loss_s_prior
        + loss_curv_anchor
        + loss_known_radius
        + loss_optic_coeff
    )
    components = {
        "total": total.detach(),
        "data": loss_data.detach(),
        "univ_gauge": weighted_univ_gauge.detach(),
        "S_gauge": weighted_S_gauge.detach(),
        "plane_l2": loss_plane_l2.detach(),
        "optic_prior": loss_optic_prior.detach(),
        "position_prior": loss_position_prior.detach(),
        "smoothness": loss_smooth.detach(),
        "seam_smoothness": loss_seam_smooth.detach(),
        "s_prior": loss_s_prior.detach(),
        "curv_anchor": loss_curv_anchor.detach(),
        "known_radius": loss_known_radius.detach(),
        "optic_coeff": loss_optic_coeff.detach(),
        "raw_univ_gauge": loss_univ_gauge.detach(),
        "raw_S_gauge": loss_S_gauge.detach(),
        "raw_smoothness": raw_smooth.detach(),
        "raw_seam_smoothness": raw_seam_smooth.detach(),
        "raw_known_radius": raw_known_radius.detach(),
    }
    return total, loss_data, components

def reconstruct_stitching(
    measurements: np.ndarray,
    masks: np.ndarray,
    scan_positions: np.ndarray,
    *,
    optic_canvas_shape: tuple[int, int] | None = None,
    fit_positions: bool = False,
    plane_order: str = "tilt",
    interp_mode: str = "bilinear",
    optic_model: str = "pixel",
    optic_poly_order: int = 10,
    optic_poly_order_x: int | None = None,
    optic_poly_order_y: int | None = None,
    optic_poly_basis: str = "legendre",
    min_output_coverage: float = 0.15,
    smoothness: float = 0.0,
    seam_smoothness: float = 0.0,
    plane_l2: float = 0.0,
    frame_batch_size: int = 16,
    device: str = "cpu",
    phase2_iters: int = 500,
    pixel_spacing: tuple[float, float] | None = None,
    known_radius_m: float | None = None,
    known_radius_axis: str = "x",
    known_radius_l2: float = 100.0,
    step_px: float | None = None,
    initial_optic: np.ndarray | None = None,
    anchor_curvature: bool = True,
    s_mode: str = "fixed_from_residual",
    s_highpass_order: int = 3,
    optic_prior_l2: float = 0.005,
    position_l2: float = 1.0,
    curv_anchor_l2: float = 100.0,
    optic_coeff_l2: float = 0.0,
    optic_coeff_degree_power: float = 4.0,
    lam_g: float = 1e-6,
    accept_tol: float = 1.05,
    rollback_factor: float = 1.5,
    lr_o: float = 1e-1,
    lr_planes: float = 1e-2,
    lr_pos: float = 1e-3,
    verbose: bool = True,
) -> StitchingResult:
    frame_shape = measurements.shape[1:]
    canvas_shape = canvas_shape_from_positions(scan_positions, frame_shape, margin=8, requested=optic_canvas_shape)
    kappa, weak_dir = scan_condition(scan_positions)
    if verbose:
        print(f"scan geometry kappa={kappa:.3g}, weak_dir={weak_dir}")
        print(f"canvas shape={canvas_shape}, frames={measurements.shape[0]}, frame shape={frame_shape}")

    fit_S_effective = False
    if s_mode not in ("fixed_from_residual", "none"):
        raise ValueError(f"unsupported s_mode {s_mode!r}")

    model = StitchingModel(
        measurements,
        masks,
        scan_positions,
        canvas_shape,
        S_known=None,
        fit_S=False,
        fit_positions=fit_positions,
        fit_power_per_frame=False,
        plane_order=plane_order,
        interp_mode=interp_mode,
        optic_model=optic_model,
        optic_poly_order=optic_poly_order,
        optic_poly_order_x=optic_poly_order_x,
        optic_poly_order_y=optic_poly_order_y,
        optic_poly_basis=optic_poly_basis,
        aperture_feather=0,
        min_output_coverage=min_output_coverage,
        pixel_spacing=pixel_spacing,
        known_radius_m=known_radius_m,
        known_radius_axis=known_radius_axis,
        step_px=step_px,
        initial_optic=initial_optic,
        anchor_curvature=anchor_curvature,
        device=device,
    )

    history: list[dict[str, Any]] = []
    fixed_systematic_summary: dict[str, Any] | None = None

    def log(
        phase: str,
        step: int,
        loss_data: torch.Tensor,
        *,
        components: dict[str, torch.Tensor] | None = None,
        accepted: bool | None = None,
        status: str | None = None,
    ) -> None:
        rms = float(torch.sqrt(loss_data.detach()).cpu().item())
        rec = {"phase": phase, "step": step, "rms": rms}
        if components is not None:
            for key, value in components.items():
                rec[f"loss_{key}"] = float(value.detach().cpu().item())
        if accepted is not None:
            rec["accepted"] = accepted
        if status is not None:
            rec["status"] = status
        if model.fit_positions:
            pos = model.current_positions().detach().cpu().numpy()
            rec["position_rms_delta"] = float(np.sqrt(np.mean((pos - scan_positions) ** 2)))
        history.append(rec)
        if verbose:
            status_text = f" {status}" if status else (" accepted" if accepted is True else (" restored" if accepted is False else ""))
            total_text = ""
            if components is not None and "total" in components:
                total_text = f", total={float(components['total'].detach().cpu().item()):.6g}"
            print(f"  {phase} {step}: RMS={rms:.6g}{total_text}{status_text}")

    def data_loss_only() -> torch.Tensor:
        _, loss_data, _ = batched_loss(
            model,
            frame_batch_size,
            lam_g=0.0,
            lam_s=0.0,
            smoothness=0.0,
            seam_smoothness=0.0,
            plane_l2=0.0,
            s_l2=0.0,
            s_smoothness=0.0,
            optic_prior_l2=0.0,
            position_l2=0.0,
            curv_anchor_l2=0.0,
            known_radius_l2=0.0,
            optic_coeff_l2=0.0,
            optic_coeff_degree_power=optic_coeff_degree_power,
        )
        return loss_data.detach()

    def optimization_loss(use_smoothness: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return batched_loss(
            model,
            frame_batch_size,
            lam_g=lam_g,
            lam_s=0.0,
            smoothness=use_smoothness,
            seam_smoothness=seam_smoothness,
            plane_l2=plane_l2,
            s_l2=0.0,
            s_smoothness=0.0,
            optic_prior_l2=optic_prior_l2,
            position_l2=position_l2,
            curv_anchor_l2=curv_anchor_l2 if model.curvature_anchor_enabled else 0.0,
            known_radius_l2=known_radius_l2 if model.known_radius_enabled else 0.0,
            optic_coeff_l2=optic_coeff_l2,
            optic_coeff_degree_power=optic_coeff_degree_power,
        )

    def snapshot_state() -> dict[str, torch.Tensor | None]:
        return {
            "O": model.O.detach().clone() if model.O is not None else None,
            "O_coeff": model.O_coeff.detach().clone() if model.O_coeff is not None else None,
            "planes": model.planes.detach().clone(),
            "S": model.S.detach().clone() if model.S is not None else None,
            "delta_pos": model.delta_pos.detach().clone() if model.delta_pos is not None else None,
        }

    def restore_state(state: dict[str, torch.Tensor | None]) -> None:
        with torch.no_grad():
            if model.O is not None and state["O"] is not None:
                model.O.copy_(state["O"])
            if model.O_coeff is not None and state["O_coeff"] is not None:
                model.O_coeff.copy_(state["O_coeff"])
            model.planes.copy_(state["planes"])
            if model.S is not None and state["S"] is not None:
                model.S.copy_(state["S"])
            if model.delta_pos is not None and state["delta_pos"] is not None:
                model.delta_pos.copy_(state["delta_pos"])

    def accept_or_restore(
        candidate_loss: torch.Tensor,
        best_loss: torch.Tensor,
        best_state: dict[str, torch.Tensor | None],
        opt: torch.optim.Optimizer | None = None,
        best_opt_state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], dict[str, Any] | None, bool | None, str]:
        candidate = float(candidate_loss.detach().cpu().item())
        best = float(best_loss.detach().cpu().item())
        if candidate <= best * accept_tol:
            opt_state = copy.deepcopy(opt.state_dict()) if opt is not None else best_opt_state
            return candidate_loss.detach(), snapshot_state(), opt_state, True, "accepted"
        if candidate > best * rollback_factor:
            restore_state(best_state)
            if opt is not None and best_opt_state is not None:
                opt.load_state_dict(best_opt_state)
            return best_loss, best_state, best_opt_state, False, "restored"
        return best_loss, best_state, best_opt_state, None, "kept"

    initial_loss, initial_loss_data, initial_components = batched_loss(
        model,
        frame_batch_size,
        lam_g=0.0,
        lam_s=0.0,
        smoothness=0.0,
        seam_smoothness=0.0,
        plane_l2=0.0,
        s_l2=0.0,
        s_smoothness=0.0,
        optic_prior_l2=0.0,
        position_l2=0.0,
        curv_anchor_l2=0.0,
        known_radius_l2=0.0,
        optic_coeff_l2=0.0,
        optic_coeff_degree_power=optic_coeff_degree_power,
    )
    log("initial", 0, initial_loss_data, components=initial_components)

    if s_mode == "fixed_from_residual":
        model.initialize_planes_from_current_optic()
        fixed_s, fixed_systematic_summary = estimate_detector_systematic_from_residual(
            model,
            frame_batch_size,
            highpass_order=s_highpass_order,
        )
        model.S_fixed = torch.as_tensor(fixed_s, dtype=torch.float64, device=model.M.device)
        model.initialize_planes_from_current_optic()

    best_loss, best_loss_data, best_components = optimization_loss(0.0)
    best_loss = best_loss.detach()
    best_state = snapshot_state()

    # Phase 2: joint Adam.
    model.project_mean_plane_to_optic()
    best_loss, best_loss_data, best_components = optimization_loss(smoothness)
    best_loss = best_loss.detach()
    best_state = snapshot_state()
    for param in model.optic_parameters():
        param.requires_grad_(True)
    params = [{"params": model.optic_parameters(), "lr": lr_o}, {"params": [model.planes], "lr": lr_planes}]
    if fit_positions:
        params.append({"params": [model.delta_pos], "lr": lr_pos})
    opt = torch.optim.Adam(params)
    best_opt_state = copy.deepcopy(opt.state_dict())
    for step in range(max(0, phase2_iters)):
        opt.zero_grad(set_to_none=True)
        loss, loss_data, components = optimization_loss(smoothness)
        loss.backward()
        opt.step()
        model.project_position_gauge()
        if phase2_iters > 0 and step + 1 == max(1, phase2_iters // 2):
            model.project_mean_plane_to_optic()
        candidate_loss, candidate_loss_data, candidate_components = optimization_loss(smoothness)
        best_loss, best_state, best_opt_state, accepted, status = accept_or_restore(
            candidate_loss.detach(), best_loss, best_state, opt, best_opt_state
        )
        if step == 0 or (step + 1) % 50 == 0 or step == phase2_iters - 1:
            log("phase2", step + 1, candidate_loss_data.detach(), components=candidate_components, accepted=accepted, status=status)

    restore_state(best_state)
    model.project_mean_plane_to_optic()
    model.project_position_gauge()
    final_optic_map = model.current_optic_map()
    final_curvature_coeff = model._optic_poly_coeff(final_optic_map).detach()
    known_radius_summary = None
    if model.known_radius_enabled and model.known_radius_target_q is not None:
        current_q, current_coeff = model.known_radius_current_q(final_optic_map)
        current_q_value = float(current_q.detach().cpu().item())
        target_q_value = float(model.known_radius_target_q.detach().cpu().item())
        known_radius_summary = {
            "enabled": True,
            "axis": model.known_radius_axis,
            "target_radius_m": model.known_radius_m,
            "target_q_nm_per_mm2": target_q_value,
            "current_q_nm_per_mm2": current_q_value,
            "current_radius_m": float(500.0 / current_q_value) if abs(current_q_value) > 1e-30 else float("nan"),
            "q_error_nm_per_mm2": current_q_value - target_q_value,
            "known_radius_l2": known_radius_l2,
            "fit_coefficients_c_bx_by_qxx_qxy_qyy": current_coeff.detach().cpu().numpy().tolist(),
        }
    residuals, reconstructed = evaluate_model(model, frame_batch_size)
    optic = final_optic_map.detach().cpu().numpy().astype(np.float64)
    observed = model.optic_observed_mask.detach().cpu().numpy().astype(np.float64)
    quality = model.optic_quality_mask.detach().cpu().numpy().astype(np.float64)
    optic[quality <= min_output_coverage] = np.nan
    s_eff = model.S_effective
    if fit_S_effective and s_eff is not None:
        systematic = s_eff.detach().cpu().numpy().astype(np.float64)
    elif model.S_fixed is not None and s_mode == "fixed_from_residual":
        systematic = model.S_fixed.detach().cpu().numpy().astype(np.float64)
    else:
        systematic = None
    planes = model.planes.detach().cpu().numpy().astype(np.float64)
    positions = model.current_positions().detach().cpu().numpy().astype(np.float64)
    summary = {
        "final_rms": finite_rms(residuals),
        "optic_rms": finite_rms(optic),
        "systematic_rms": finite_rms(systematic) if systematic is not None else None,
        "scan_kappa": kappa,
        "scan_is_1d": model.scan_is_1d,
        "max_delta_pos": model.max_delta_pos,
        "fit_S": False,
        "s_mode": s_mode,
        "s_highpass_order": s_highpass_order,
        "systematic_init_summary": None,
        "fixed_systematic_summary": fixed_systematic_summary,
        "fit_positions": fit_positions,
        "fit_power_per_frame": False,
        "plane_order": plane_order,
        "initial_optic": "sequential" if initial_optic is not None else "warm_average",
        "optic_model": model.optic_model,
        "optic_poly_order": optic_poly_order,
        "optic_poly_order_x": optic_poly_order_x,
        "optic_poly_order_y": optic_poly_order_y,
        "optic_poly_basis": optic_poly_basis,
        "optic_poly_terms": [list(term) for term in model.O_terms] if model.O_terms is not None else None,
        "optic_poly_coefficients": model.O_coeff.detach().cpu().numpy().tolist() if model.O_coeff is not None else None,
        "curvature_anchor": bool(model.curvature_anchor_enabled),
        "curvature_anchor_coeff_normalized": model.curvature_prior_coeff.detach().cpu().numpy().tolist() if model.curvature_anchor_enabled else None,
        "curvature_final_coeff_normalized": final_curvature_coeff.cpu().numpy().tolist(),
        "curvature_anchor_delta_normalized": (final_curvature_coeff - model.curvature_prior_coeff).cpu().numpy().tolist() if model.curvature_anchor_enabled else None,
        "optic_prior_l2": optic_prior_l2,
        "position_l2": position_l2,
        "curv_anchor_l2": curv_anchor_l2,
        "known_radius_constraint": known_radius_summary if known_radius_summary is not None else {"enabled": False},
        "optic_coeff_l2": optic_coeff_l2,
        "optic_coeff_degree_power": optic_coeff_degree_power,
        "lam_g": lam_g,
        "accept_tol": accept_tol,
        "rollback_factor": rollback_factor,
        "position_gauge": "1d_zero_y_and_remove_x_affine" if model.scan_is_1d and fit_positions else ("remove_mean_xy" if fit_positions else None),
        "position_delta_rms_px": float(np.sqrt(np.mean((positions - scan_positions) ** 2))) if fit_positions else 0.0,
        "position_delta_pv_px": finite_pv(positions - scan_positions) if fit_positions else 0.0,
        "interp_mode": interp_mode,
        "aperture_feather": 0,
        "min_output_coverage": min_output_coverage,
        "plane_l2": plane_l2,
        "smoothness": smoothness,
        "seam_smoothness": seam_smoothness,
    }
    return StitchingResult(optic, systematic, planes, positions, residuals, reconstructed, observed, quality, history, summary)


def estimate_detector_systematic_from_residual(
    model: StitchingModel,
    frame_batch_size: int,
    *,
    highpass_order: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual_stack = []
    with torch.no_grad():
        for start in range(0, model.N, frame_batch_size):
            idx = torch.arange(start, min(model.N, start + frame_batch_size), device=model.M.device)
            pred, mask, frame_idx = model.forward_batch(idx)
            res = model.M[frame_idx] - pred
            res_np = res.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy() > 0
            res_np[~mask_np] = np.nan
            residual_stack.append(res_np)
    residual_stack_np = np.concatenate(residual_stack, axis=0)
    finite_count = np.isfinite(residual_stack_np).sum(axis=0)
    systematic = np.zeros(residual_stack_np.shape[1:], dtype=np.float64)
    valid_any = finite_count > 0
    with np.errstate(invalid="ignore"):
        systematic[valid_any] = np.nanmedian(residual_stack_np[:, valid_any], axis=0)
    valid = np.isfinite(systematic)
    systematic_hp, coeff, terms = remove_low_order_legendre(
        systematic,
        order=highpass_order,
        weight=valid.astype(np.float64),
    )
    systematic_hp[~valid] = 0.0
    summary = {
        "method": "median detector-coordinate residual, with low-order Legendre detector modes removed",
        "highpass_order": highpass_order,
        "low_order_terms": [list(term) for term in terms],
        "low_order_coefficients": coeff.tolist(),
        "systematic_rms_nm": finite_rms(systematic_hp),
        "systematic_pv_nm": finite_pv(systematic_hp),
        "valid_pixels": int(np.count_nonzero(valid)),
    }
    return systematic_hp.astype(np.float64), summary


def evaluate_model(model: StitchingModel, frame_batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.full(tuple(model.M.shape), np.nan, dtype=np.float64)
    reconstructed = np.full(tuple(model.M.shape), np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, model.N, frame_batch_size):
            idx = torch.arange(start, min(model.N, start + frame_batch_size), device=model.M.device)
            pred, mask, frame_idx = model.forward_batch(idx)
            res = (model.M[frame_idx] - pred).detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy() > 0.0
            for local, global_idx in enumerate(frame_idx.detach().cpu().numpy()):
                residuals[global_idx][mask_np[local]] = res[local][mask_np[local]]
                reconstructed[global_idx][mask_np[local]] = pred_np[local][mask_np[local]]
    return residuals, reconstructed


def align_offset(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.nanmean(a[valid] - b[valid])) if np.any(valid) else 0.0


def aligned_difference_remove_piston_tilt(
    a: np.ndarray,
    b: np.ndarray,
    pixel_spacing: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = a.shape
    sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * sy_mm
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * sx_mm
    yy, xx = np.meshgrid(y, x, indexing="ij")
    valid = np.isfinite(a) & np.isfinite(b)
    diff = np.full_like(a, np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 3:
        return diff, {"error": "not enough common pixels", "valid_pixels": int(np.count_nonzero(valid))}
    raw = a - b
    design = np.column_stack([np.ones(np.count_nonzero(valid)), xx[valid], yy[valid]])
    coeff, *_ = np.linalg.lstsq(design, raw[valid], rcond=None)
    trend = coeff[0] + coeff[1] * xx + coeff[2] * yy
    diff[valid] = raw[valid] - trend[valid]
    return diff, {
        "model": "difference_nm = c + bx*x_mm + by*y_mm + residual; c/bx/by removed",
        "coeff_names": ["c_nm", "bx_nm_per_mm", "by_nm_per_mm"],
        "coefficients": [float(v) for v in coeff],
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def finite_pv(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.nanmax(values[valid]) - np.nanmin(values[valid])) if np.any(valid) else float("nan")


def finite_to_float32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


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


def pixel_spacing_from_data(data: dict[str, Any]) -> tuple[float, float] | None:
    if "raw_pixel_spacing_mm" not in data:
        return None
    try:
        sx = float(np.asarray(data["raw_pixel_spacing_mm"]).reshape(-1)[0])
        sy = float(np.asarray(data.get("raw_pixel_spacing_y_mm", data["raw_pixel_spacing_mm"])).reshape(-1)[0])
        downsample = float(np.asarray(data.get("raw_downsample", 1.0)).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    if sx <= 0 or sy <= 0 or downsample <= 0:
        return None
    return sy * downsample, sx * downsample


def profile_axis_nanmedian(image: np.ndarray, axis: int) -> np.ndarray:
    """Median height profile while keeping all-NaN rows/columns as NaN."""

    arr = np.asarray(image, dtype=np.float64)
    out_len = arr.shape[1] if axis == 0 else arr.shape[0]
    profile = np.full(out_len, np.nan, dtype=np.float64)
    if axis == 0:
        for idx in range(arr.shape[1]):
            vals = arr[:, idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                profile[idx] = float(np.median(vals))
    elif axis == 1:
        for idx in range(arr.shape[0]):
            vals = arr[idx, :]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                profile[idx] = float(np.median(vals))
    else:
        raise ValueError("axis must be 0 or 1")
    return profile


def map_axes_mm(shape: tuple[int, int], pixel_spacing: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
    y_mm = (np.arange(shape[0], dtype=np.float64) - 0.5 * (shape[0] - 1)) * sy_mm
    x_mm = (np.arange(shape[1], dtype=np.float64) - 0.5 * (shape[1] - 1)) * sx_mm
    return y_mm, x_mm


def add_second_order_npz_outputs(
    output_arrays: dict[str, Any],
    prefix: str,
    raw_height: np.ndarray,
    fitted_height: np.ndarray,
    residual_height: np.ndarray,
    fit_summary: dict[str, Any],
    pixel_spacing: tuple[float, float] | None,
) -> None:
    """Store raw/fitted/residual maps, 1-D profiles, coefficients, and radii."""

    y_mm, x_mm = map_axes_mm(raw_height.shape, pixel_spacing)
    output_arrays[f"{prefix}_raw_height_nm"] = raw_height
    output_arrays[f"{prefix}_second_order_fit_height_nm"] = fitted_height
    output_arrays[f"{prefix}_second_order_residual_height_nm"] = residual_height
    output_arrays[f"{prefix}_profile_x_mm"] = x_mm
    output_arrays[f"{prefix}_profile_y_mm"] = y_mm
    output_arrays[f"{prefix}_raw_profile_x_nm"] = profile_axis_nanmedian(raw_height, axis=0)
    output_arrays[f"{prefix}_raw_profile_y_nm"] = profile_axis_nanmedian(raw_height, axis=1)
    output_arrays[f"{prefix}_fit_profile_x_nm"] = profile_axis_nanmedian(fitted_height, axis=0)
    output_arrays[f"{prefix}_fit_profile_y_nm"] = profile_axis_nanmedian(fitted_height, axis=1)
    output_arrays[f"{prefix}_residual_profile_x_nm"] = profile_axis_nanmedian(residual_height, axis=0)
    output_arrays[f"{prefix}_residual_profile_y_nm"] = profile_axis_nanmedian(residual_height, axis=1)
    output_arrays[f"{prefix}_second_order_coefficients"] = np.asarray(fit_summary.get("coefficients", []), dtype=np.float64)
    output_arrays[f"{prefix}_second_order_coeff_names"] = np.asarray(fit_summary.get("coeff_names", []), dtype=object)
    output_arrays[f"{prefix}_radius_m"] = np.asarray(
        [fit_summary.get("radius_x_m", np.nan), fit_summary.get("radius_y_m", np.nan)],
        dtype=np.float64,
    )
    output_arrays[f"{prefix}_radius_names"] = np.asarray(["Rx_m", "Ry_m"], dtype=object)
    output_arrays[f"{prefix}_principal_radius_m"] = np.asarray(fit_summary.get("principal_radius_m", []), dtype=np.float64)
    output_arrays[f"{prefix}_principal_q_nm_per_mm2"] = np.asarray(fit_summary.get("principal_q_nm_per_mm2", []), dtype=np.float64)
    output_arrays[f"{prefix}_principal_axis_vectors_xy"] = np.asarray(fit_summary.get("principal_axis_vectors_xy", []), dtype=np.float64)
    output_arrays[f"{prefix}_principal_axis_angles_deg_from_x"] = np.asarray(fit_summary.get("principal_axis_angles_deg_from_x", []), dtype=np.float64)
    output_arrays[f"{prefix}_quadratic_matrix_nm_per_mm2"] = np.asarray(fit_summary.get("quadratic_matrix_nm_per_mm2", []), dtype=np.float64)
    output_arrays[f"{prefix}_second_order_residual_rms_nm"] = np.asarray(fit_summary.get("residual_rms_nm", np.nan), dtype=np.float64)
    output_arrays[f"{prefix}_second_order_residual_pv_nm"] = np.asarray(fit_summary.get("residual_pv_nm", np.nan), dtype=np.float64)
    output_arrays[f"{prefix}_second_order_summary_json"] = np.asarray(json.dumps(fit_summary), dtype=object)


def fit_second_order_surface(
    optic: np.ndarray,
    observed_mask: np.ndarray,
    pixel_spacing: tuple[float, float] | None,
    *,
    include_twist: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = optic.shape
    sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * sy_mm
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * sx_mm
    yy, xx = np.meshgrid(y, x, indexing="ij")
    valid = np.isfinite(optic) & np.isfinite(observed_mask) & (observed_mask > 0)
    if int(np.count_nonzero(valid)) < 5:
        fitted = np.full_like(optic, np.nan, dtype=np.float64)
        residual = np.full_like(optic, np.nan, dtype=np.float64)
        return fitted, residual, {"valid_pixels": int(np.count_nonzero(valid)), "error": "not enough valid pixels"}

    if include_twist:
        design = np.column_stack([
            np.ones(np.count_nonzero(valid), dtype=np.float64),
            xx[valid],
            yy[valid],
            xx[valid] ** 2,
            xx[valid] * yy[valid],
            yy[valid] ** 2,
        ])
        coeff_fit, *_ = np.linalg.lstsq(design, optic[valid], rcond=None)
        coeff = coeff_fit
        full_design = np.stack([np.ones_like(xx), xx, yy, xx**2, xx * yy, yy**2], axis=0)
    else:
        design = np.column_stack([
            np.ones(np.count_nonzero(valid), dtype=np.float64),
            xx[valid],
            yy[valid],
            xx[valid] ** 2,
            yy[valid] ** 2,
        ])
        coeff_fit, *_ = np.linalg.lstsq(design, optic[valid], rcond=None)
        coeff = np.asarray([coeff_fit[0], coeff_fit[1], coeff_fit[2], coeff_fit[3], 0.0, coeff_fit[4]], dtype=np.float64)
        full_design = np.stack([np.ones_like(xx), xx, yy, xx**2, xx * yy, yy**2], axis=0)
    fitted = np.tensordot(coeff, full_design, axes=(0, 0))
    residual = optic - fitted
    residual[~valid] = np.nan

    qxx = float(coeff[3])
    qxy = float(coeff[4])
    qyy = float(coeff[5])
    quad = np.array([[qxx, 0.5 * qxy], [0.5 * qxy, qyy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(quad)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    principal_radii = np.asarray([500.0 / q if abs(q) > 1e-30 else np.nan for q in eigvals], dtype=np.float64)
    principal_angles_deg = np.degrees(np.arctan2(eigvecs[1, :], eigvecs[0, :]))
    radius_x = float(500.0 / qxx) if abs(qxx) > 1e-30 else float("nan")
    radius_y = float(500.0 / qyy) if abs(qyy) > 1e-30 else float("nan")
    summary = {
        "model": "height_nm = c + bx*x_mm + by*y_mm + qxx*x_mm^2 + qxy*x_mm*y_mm + qyy*y_mm^2" if include_twist else "height_nm = c + bx*x_mm + by*y_mm + qxx*x_mm^2 + qyy*y_mm^2; qxy fixed to 0",
        "include_twist": bool(include_twist),
        "sag_convention": "height_m = u_m^2/(2*R_m), so R_m = 500/q for q in nm/mm^2 along a principal axis",
        "rotation_note": "The x*y twist term is included in this fit." if include_twist else "The x*y twist term is fixed to zero in this fit; twist remains in the residual.",
        "pixel_spacing_y_mm": sy_mm,
        "pixel_spacing_x_mm": sx_mm,
        "valid_pixels": int(np.count_nonzero(valid)),
        "coeff_names": ["c_nm", "bx_nm_per_mm", "by_nm_per_mm", "qxx_nm_per_mm2", "qxy_nm_per_mm2", "qyy_nm_per_mm2"],
        "coefficients": [float(v) for v in coeff],
        "quadratic_matrix_nm_per_mm2": quad.tolist(),
        "principal_q_nm_per_mm2": eigvals.tolist(),
        "principal_radius_m": principal_radii.tolist(),
        "principal_axis_vectors_xy": eigvecs.T.tolist(),
        "principal_axis_angles_deg_from_x": principal_angles_deg.tolist(),
        "radius_x_m": radius_x,
        "radius_y_m": radius_y,
        "radius_major_m": float(principal_radii[0]),
        "radius_minor_m": float(principal_radii[1]) if principal_radii.size > 1 else float("nan"),
        "residual_rms_nm": finite_rms(residual),
        "residual_pv_nm": finite_pv(residual),
    }
    return fitted, residual, summary


def remove_global_tilt(
    image: np.ndarray,
    pixel_spacing: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape
    sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * sy_mm
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * sx_mm
    yy, xx = np.meshgrid(y, x, indexing="ij")
    valid = np.isfinite(image)
    if np.count_nonzero(valid) < 3:
        return image.copy(), {"error": "not enough valid pixels", "valid_pixels": int(np.count_nonzero(valid))}
    design = np.column_stack([np.ones(np.count_nonzero(valid)), xx[valid], yy[valid]])
    coeff, *_ = np.linalg.lstsq(design, image[valid], rcond=None)
    tilt = coeff[1] * xx + coeff[2] * yy
    corrected = image - tilt - coeff[0]
    corrected[~valid] = np.nan
    return corrected, {
        "model": "height_nm = c + bx*x_mm + by*y_mm; only bx/by tilt is removed",
        "coeff_names": ["c_nm", "bx_nm_per_mm", "by_nm_per_mm"],
        "coefficients": [float(v) for v in coeff],
        "valid_pixels": int(np.count_nonzero(valid)),
    }


def local_second_order_statistics(
    measurements: np.ndarray,
    masks: np.ndarray,
    pixel_spacing: tuple[float, float] | None,
) -> dict[str, Any]:
    n, h, w = measurements.shape
    sy_mm, sx_mm = pixel_spacing if pixel_spacing is not None else (1.0, 1.0)
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * sy_mm
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * sx_mm
    yy, xx = np.meshgrid(y, x, indexing="ij")
    coeffs = []
    radii = []
    for index in range(n):
        im = measurements[index]
        valid = np.isfinite(im) & (masks[index] > 0)
        if np.count_nonzero(valid) < 5:
            coeffs.append([np.nan, np.nan, np.nan, np.nan, np.nan])
            radii.append([np.nan, np.nan])
            continue
        design = np.column_stack([
            np.ones(np.count_nonzero(valid), dtype=np.float64),
            xx[valid],
            yy[valid],
            xx[valid] ** 2,
            yy[valid] ** 2,
        ])
        coeff, *_ = np.linalg.lstsq(design, im[valid], rcond=None)
        qxx = float(coeff[3])
        qyy = float(coeff[4])
        coeffs.append([float(v) for v in coeff])
        radii.append([
            float(500.0 / qxx) if abs(qxx) > 1e-30 else float("nan"),
            float(500.0 / qyy) if abs(qyy) > 1e-30 else float("nan"),
        ])
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    return {
        "model": "per-frame height_nm = c + bx*x_mm + by*y_mm + qxx*x_mm^2 + qyy*y_mm^2",
        "coeff_names": ["c_nm", "bx_nm_per_mm", "by_nm_per_mm", "qxx_nm_per_mm2", "qyy_nm_per_mm2"],
        "coefficient_mean": np.nanmean(coeffs_arr, axis=0).tolist(),
        "coefficient_std": np.nanstd(coeffs_arr, axis=0).tolist(),
        "radius_names": ["Rx_m", "Ry_m"],
        "radius_median_m": np.nanmedian(radii_arr, axis=0).tolist(),
        "radius_mean_m": np.nanmean(radii_arr, axis=0).tolist(),
        "radius_std_m": np.nanstd(radii_arr, axis=0).tolist(),
        "radius_pv_m": (np.nanmax(radii_arr, axis=0) - np.nanmin(radii_arr, axis=0)).tolist(),
    }


def sequential_overlap_stitch(
    measurements: np.ndarray,
    masks: np.ndarray,
    positions: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n, h, w = measurements.shape
    H, W = canvas_shape
    numerator = np.zeros((H, W), dtype=np.float64)
    denominator = np.zeros((H, W), dtype=np.float64)
    plane_coeffs = np.zeros((n, 3), dtype=np.float64)
    yy_local, xx_local = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )

    for index in range(n):
        py, px = np.round(positions[index]).astype(int)
        patch = measurements[index].astype(np.float64, copy=True)
        valid = np.isfinite(patch) & (masks[index] > 0)
        y0 = max(0, py)
        x0 = max(0, px)
        y1 = min(H, py + h)
        x1 = min(W, px + w)
        if y1 <= y0 or x1 <= x0:
            plane_coeffs[index] = np.nan
            continue
        sy0 = y0 - py
        sx0 = x0 - px
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)
        local_valid = valid[sy0:sy1, sx0:sx1]
        existing_valid = denominator[y0:y1, x0:x1] > 0
        overlap = local_valid & existing_valid

        if np.count_nonzero(overlap) >= 3:
            existing = numerator[y0:y1, x0:x1][overlap] / denominator[y0:y1, x0:x1][overlap]
            target = existing - patch[sy0:sy1, sx0:sx1][overlap]
            design = np.column_stack([
                np.ones(np.count_nonzero(overlap), dtype=np.float64),
                xx_local[sy0:sy1, sx0:sx1][overlap],
                yy_local[sy0:sy1, sx0:sx1][overlap],
            ])
            coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
        elif np.any(valid):
            coeff = np.array([-float(np.nanmedian(patch[valid])), 0.0, 0.0], dtype=np.float64)
        else:
            coeff = np.zeros(3, dtype=np.float64)

        plane_coeffs[index] = coeff
        patch = patch + coeff[0] + coeff[1] * xx_local + coeff[2] * yy_local
        local_weight = masks[index][sy0:sy1, sx0:sx1].astype(np.float64, copy=False)
        add = local_valid & np.isfinite(patch[sy0:sy1, sx0:sx1])
        numerator[y0:y1, x0:x1][add] += patch[sy0:sy1, sx0:sx1][add] * local_weight[add]
        denominator[y0:y1, x0:x1][add] += local_weight[add]

    stitched = np.full((H, W), np.nan, dtype=np.float64)
    valid_canvas = denominator > 0
    stitched[valid_canvas] = numerator[valid_canvas] / denominator[valid_canvas]
    return stitched, plane_coeffs


def save_map_figure(path: Path, image: np.ndarray, title: str, *, cmap: str = "viridis", symmetric: bool = True) -> None:
    import matplotlib.pyplot as plt

    vals = image[np.isfinite(image)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    elif symmetric:
        lim = np.percentile(np.abs(vals), 99)
        if not np.isfinite(lim) or lim <= 0:
            lim = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_figures(result: StitchingResult, data: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    meas = data["measurements"]
    n = meas.shape[0]
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    def grid_plot(stack: np.ndarray, title: str, filename: str) -> None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True)
        vals = stack[np.isfinite(stack)]
        lim = np.percentile(np.abs(vals), 99) if vals.size else 1
        for ax in axes.ravel():
            ax.axis("off")
        for i in range(n):
            ax = axes.ravel()[i]
            im = ax.imshow(stack[i], origin="lower", cmap="viridis", vmin=-lim, vmax=lim)
            ax.set_title(f"frame {i}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    grid_plot(meas, "simulated measurements", "measurements.png")
    grid_plot(result.reconstructed, "reconstructed measurements", "reconstructed_measurements.png")
    grid_plot(result.residuals, "residuals", "residuals.png")

    maps = [result.optic]
    titles = ["recovered optic O"]
    maps.append(result.observed_mask)
    titles.append("normalized optic coverage")
    if "optic_true" in data:
        true = data["optic_true"]
        off = align_offset(result.optic, true)
        maps += [true, result.optic - true - off]
        titles += ["true optic", "optic error"]
    if result.systematic is not None:
        maps.append(result.systematic)
        titles.append("recovered S")
        if "systematic_true" in data:
            s_true = data["systematic_true"]
            off = align_offset(result.systematic, s_true)
            maps += [s_true, result.systematic - s_true - off]
            titles += ["true S", "S error"]

    cols = min(3, len(maps))
    rows = int(np.ceil(len(maps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, imdata, title in zip(axes.ravel(), maps, titles):
        vals = imdata[np.isfinite(imdata)]
        if "coverage" in title:
            vmin, vmax = 0.0, 1.0
        else:
            lim = np.percentile(np.abs(vals), 99) if vals.size else 1
            vmin, vmax = -lim, lim
        im = ax.imshow(imdata, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "recovered_maps.png", dpi=160)
    plt.close(fig)

    if result.loss_history:
        x = np.arange(len(result.loss_history))
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        component_keys = [
            "loss_total",
            "loss_data",
            "loss_optic_prior",
            "loss_plane_l2",
            "loss_curv_anchor",
            "loss_known_radius",
            "loss_optic_coeff",
            "loss_s_prior",
            "loss_smoothness",
            "loss_seam_smoothness",
            "loss_position_prior",
            "loss_univ_gauge",
        ]
        for key in component_keys:
            y = np.array([float(r.get(key, np.nan)) for r in result.loss_history], dtype=np.float64)
            valid = np.isfinite(y) & (y > 0)
            if np.any(valid):
                ax.semilogy(x[valid], y[valid], marker="o", label=key.replace("loss_", ""))
        ax.set_xlabel("logged step")
        ax.set_ylabel("loss value (nm^2 or weighted term)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
        fig.savefig(output_dir / "loss_history.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot([r["rms"] for r in result.loss_history], marker="o")
        ax.set_xlabel("logged step")
        ax.set_ylabel("RMS residual (nm)")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "rms_history.png", dpi=160)
        plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def second_order_qxy(fit_summary: Any) -> float | None:
    if not isinstance(fit_summary, dict):
        return None
    coeff = fit_summary.get("coefficients")
    if isinstance(coeff, dict):
        value = coeff.get("qxy")
        return float(value) if value is not None else None
    if isinstance(coeff, (list, tuple)) and len(coeff) > 4:
        return float(coeff[4])
    return None


def write_run_parameters(path: Path, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "script": Path(__file__).name,
        "script_path": str(Path(__file__).resolve()),
        "command_line": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "parameters": vars(args),
        "resolved_paths": {
            "input": str(Path(args.input).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
        },
        "environment": {
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
            "MPLBACKEND": os.environ.get("MPLBACKEND"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        },
    }
    if extra:
        record.update(extra)
    path.write_text(json.dumps(json_safe(record), indent=2, ensure_ascii=False), encoding="utf-8")


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}





def run(args: argparse.Namespace) -> None:
    data = load_npz(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_parameters_path = output_dir / "run_parameters.json"
    write_run_parameters(run_parameters_path, args, {"status": "started"})
    pixel_spacing = pixel_spacing_from_data(data)
    canvas_shape = tuple(data["optic_shape"]) if "optic_shape" in data else canvas_shape_from_positions(
        data["scan_positions"], data["measurements"].shape[1:], margin=8, requested=None
    )

    save_tiff(output_dir / "measurements_stack.tiff", data["measurements"])

    output_arrays: dict[str, Any] = {"raw_files": data.get("raw_files", np.array([]))}
    summary: dict[str, Any] = {
        "mode": args.mode,
        "adp_init": args.adp_init,
        "adp_curvature_anchor": args.adp_curvature_anchor,
        "input": str(Path(args.input)),
        "input_local_second_order": local_second_order_statistics(data["measurements"], data["masks"], pixel_spacing),
        "aperture_feather_effective": 0,
        "aperture_feather_note": "aperture feathering is disabled because it introduced stripe artifacts in recovered optics",
    }

    need_sequential = args.mode in ("sequential", "both") or (args.mode == "autodiff" and args.adp_init == "sequential")
    save_sequential = args.mode in ("sequential", "both")
    sequential_optic_raw: np.ndarray | None = None
    sequential_optic: np.ndarray | None = None
    sequential_plane_coeffs: np.ndarray | None = None

    if need_sequential:
        sequential_optic_raw, sequential_plane_coeffs = sequential_overlap_stitch(
            data["measurements"],
            data["masks"],
            data["scan_positions"],
            canvas_shape,
        )
        sequential_optic, sequential_tilt_summary = remove_global_tilt(sequential_optic_raw, pixel_spacing)
        sequential_second_order_fit, sequential_second_order_residual, sequential_second_order_summary = fit_second_order_surface(
            sequential_optic,
            np.isfinite(sequential_optic).astype(np.float64),
            pixel_spacing,
            include_twist=True,
        )
        sequential_second_order_fit_no_twist, sequential_second_order_residual_no_twist, sequential_second_order_summary_no_twist = fit_second_order_surface(
            sequential_optic,
            np.isfinite(sequential_optic).astype(np.float64),
            pixel_spacing,
            include_twist=False,
        )
        summary["sequential_global_tilt_removed"] = sequential_tilt_summary
        summary["sequential_second_order_fit"] = sequential_second_order_summary
        summary["sequential_second_order_fit_with_twist"] = sequential_second_order_summary
        summary["sequential_second_order_fit_no_twist"] = sequential_second_order_summary_no_twist
        summary["sequential_overlap_baseline"] = {
            "method": "frames placed at preprocessed scan positions; each new frame piston/x-tilt/y-tilt is least-squares fitted on the overlap with the current canvas; overlapping pixels are weighted-averaged; final global tilt is removed",
            "position_source": "data['scan_positions']",
            "integer_position_rounding": True,
            "plane_coeff_rms_nm": finite_rms(sequential_plane_coeffs),
        }
        output_arrays.update({
            "sequential_overlap_optic_raw": sequential_optic_raw,
            "sequential_overlap_optic": sequential_optic,
            "sequential_plane_coeffs": sequential_plane_coeffs,
            "sequential_second_order_fit": sequential_second_order_fit,
            "sequential_second_order_residual": sequential_second_order_residual,
            "sequential_second_order_fit_with_twist": sequential_second_order_fit,
            "sequential_second_order_residual_with_twist": sequential_second_order_residual,
            "sequential_second_order_fit_no_twist": sequential_second_order_fit_no_twist,
            "sequential_second_order_residual_no_twist": sequential_second_order_residual_no_twist,
        })
        add_second_order_npz_outputs(
            output_arrays,
            "baseline",
            sequential_optic,
            sequential_second_order_fit,
            sequential_second_order_residual,
            sequential_second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "sequential",
            sequential_optic,
            sequential_second_order_fit,
            sequential_second_order_residual,
            sequential_second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "baseline_with_twist",
            sequential_optic,
            sequential_second_order_fit,
            sequential_second_order_residual,
            sequential_second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "baseline_no_twist",
            sequential_optic,
            sequential_second_order_fit_no_twist,
            sequential_second_order_residual_no_twist,
            sequential_second_order_summary_no_twist,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "sequential_with_twist",
            sequential_optic,
            sequential_second_order_fit,
            sequential_second_order_residual,
            sequential_second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "sequential_no_twist",
            sequential_optic,
            sequential_second_order_fit_no_twist,
            sequential_second_order_residual_no_twist,
            sequential_second_order_summary_no_twist,
            pixel_spacing,
        )
        output_arrays["baseline_raw_height_before_global_tilt_removal_nm"] = sequential_optic_raw
        output_arrays["sequential_raw_height_before_global_tilt_removal_nm"] = sequential_optic_raw
        if save_sequential:
            save_tiff(output_dir / "sequential_overlap_stitch_raw.tiff", sequential_optic_raw)
            save_tiff(output_dir / "sequential_overlap_stitch.tiff", sequential_optic)
            save_tiff(output_dir / "sequential_second_order_fit.tiff", sequential_second_order_fit)
            save_tiff(output_dir / "sequential_second_order_residual.tiff", sequential_second_order_residual)
            save_tiff(output_dir / "sequential_second_order_fit_with_twist.tiff", sequential_second_order_fit)
            save_tiff(output_dir / "sequential_second_order_residual_with_twist.tiff", sequential_second_order_residual)
            save_tiff(output_dir / "sequential_second_order_fit_no_twist.tiff", sequential_second_order_fit_no_twist)
            save_tiff(output_dir / "sequential_second_order_residual_no_twist.tiff", sequential_second_order_residual_no_twist)
            save_map_figure(output_dir / "sequential_overlap_stitch.png", sequential_optic, "sequential overlap stitching, global tilt removed", symmetric=False)
            save_map_figure(output_dir / "sequential_second_order_residual.png", sequential_second_order_residual, "sequential baseline residual after 2nd-order fit, with twist")
            save_map_figure(output_dir / "sequential_second_order_residual_no_twist.png", sequential_second_order_residual_no_twist, "sequential baseline residual after 2nd-order fit, no twist")

    result: StitchingResult | None = None
    if args.mode in ("autodiff", "both"):
        initial_optic = sequential_optic if args.adp_init == "sequential" else None
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        result = reconstruct_stitching(
            data["measurements"],
            data["masks"],
            data["scan_positions"],
            optic_canvas_shape=canvas_shape,
            fit_positions=args.fit_positions,
            plane_order=args.plane_order,
            interp_mode=args.interp_mode,
            optic_model=args.optic_model,
            optic_poly_order=args.optic_poly_order,
            optic_poly_order_x=args.optic_poly_order_x,
            optic_poly_order_y=args.optic_poly_order_y,
            optic_poly_basis=args.optic_poly_basis,
            min_output_coverage=args.min_output_coverage,
            smoothness=args.smoothness,
            seam_smoothness=args.seam_smoothness,
            plane_l2=args.plane_l2,
            frame_batch_size=args.batch_size,
            device=device,
            phase2_iters=args.phase2_iters,
            pixel_spacing=pixel_spacing,
            known_radius_m=args.known_radius_m,
            known_radius_axis=args.known_radius_axis,
            known_radius_l2=args.known_radius_l2,
            step_px=step_px_from_data(data),
            initial_optic=initial_optic,
            anchor_curvature=args.adp_curvature_anchor == "initial",
            s_mode=args.s_mode,
            s_highpass_order=args.s_highpass_order,
            optic_prior_l2=args.adp_prior_l2,
            position_l2=args.position_l2,
            curv_anchor_l2=args.curv_anchor_l2,
            optic_coeff_l2=args.optic_coeff_l2,
            optic_coeff_degree_power=args.optic_coeff_degree_power,
            lam_g=args.lam_g,
            accept_tol=args.accept_tol,
            rollback_factor=args.rollback_factor,
            lr_o=args.lr_o,
            lr_planes=args.lr_planes,
            lr_pos=args.lr_pos,
        )

        save_tiff(output_dir / "reconstructed_stack.tiff", result.reconstructed)
        save_tiff(output_dir / "residuals_stack.tiff", result.residuals)
        save_tiff(output_dir / "binary_coverage.tiff", result.observed_mask)
        save_tiff(output_dir / "quality_coverage.tiff", result.quality_mask)
        save_tiff(output_dir / "recovered_optic.tiff", result.optic)
        save_map_figure(output_dir / "recovered_optic.png", result.optic, "ADP/autodiff recovered optic before 2nd-order fit")
        if result.systematic is not None:
            save_tiff(output_dir / "systematic_map.tiff", result.systematic)
            save_map_figure(output_dir / "systematic_map.png", result.systematic, "detector-fixed aperture systematic S")

        optic_second_order_fit, optic_second_order_residual, second_order_summary = fit_second_order_surface(
            result.optic,
            result.observed_mask,
            pixel_spacing,
            include_twist=True,
        )
        optic_second_order_fit_no_twist, optic_second_order_residual_no_twist, second_order_summary_no_twist = fit_second_order_surface(
            result.optic,
            result.observed_mask,
            pixel_spacing,
            include_twist=False,
        )
        save_tiff(output_dir / "optic_second_order_fit.tiff", optic_second_order_fit)
        save_tiff(output_dir / "optic_second_order_residual.tiff", optic_second_order_residual)
        save_tiff(output_dir / "optic_second_order_fit_with_twist.tiff", optic_second_order_fit)
        save_tiff(output_dir / "optic_second_order_residual_with_twist.tiff", optic_second_order_residual)
        save_tiff(output_dir / "optic_second_order_fit_no_twist.tiff", optic_second_order_fit_no_twist)
        save_tiff(output_dir / "optic_second_order_residual_no_twist.tiff", optic_second_order_residual_no_twist)
        save_map_figure(output_dir / "optic_second_order_residual.png", optic_second_order_residual, "ADP/autodiff optic residual after 2nd-order fit, with twist")
        save_map_figure(output_dir / "optic_second_order_residual_no_twist.png", optic_second_order_residual_no_twist, "ADP/autodiff optic residual after 2nd-order fit, no twist")
        save_figures(result, data, output_dir)

        reconstruction_error = result.residuals
        reconstruction_error_rms_by_frame = np.asarray([finite_rms(frame) for frame in reconstruction_error], dtype=np.float64)
        reconstruction_error_pv_by_frame = np.asarray([finite_pv(frame) for frame in reconstruction_error], dtype=np.float64)
        recovered_systematic = result.systematic if result.systematic is not None else np.array([], dtype=np.float64)
        output_arrays.update({
            "input_measurements_nm": data["measurements"],
            "optic": result.optic,
            "systematic": recovered_systematic,
            "recovered_systematic_error_nm": recovered_systematic,
            "recovered_systematic_profile_x_nm": profile_axis_nanmedian(recovered_systematic, axis=0) if recovered_systematic.size else np.array([], dtype=np.float64),
            "recovered_systematic_profile_y_nm": profile_axis_nanmedian(recovered_systematic, axis=1) if recovered_systematic.size else np.array([], dtype=np.float64),
            "planes": result.planes,
            "positions": result.positions,
            "residuals": reconstruction_error,
            "reconstruction_error_nm": reconstruction_error,
            "adp_reconstruction_error_nm": reconstruction_error,
            "reconstructed": result.reconstructed,
            "reconstructed_measurements_nm": result.reconstructed,
            "adp_reconstructed_measurements_nm": result.reconstructed,
            "reconstruction_error_rms_by_frame_nm": reconstruction_error_rms_by_frame,
            "adp_reconstruction_error_rms_by_frame_nm": reconstruction_error_rms_by_frame,
            "reconstruction_error_pv_by_frame_nm": reconstruction_error_pv_by_frame,
            "adp_reconstruction_error_pv_by_frame_nm": reconstruction_error_pv_by_frame,
            "reconstruction_error_global_rms_nm": np.asarray(finite_rms(reconstruction_error), dtype=np.float64),
            "adp_reconstruction_error_global_rms_nm": np.asarray(finite_rms(reconstruction_error), dtype=np.float64),
            "reconstruction_error_global_pv_nm": np.asarray(finite_pv(reconstruction_error), dtype=np.float64),
            "adp_reconstruction_error_global_pv_nm": np.asarray(finite_pv(reconstruction_error), dtype=np.float64),
            "observed_mask": result.observed_mask,
            "quality_mask": result.quality_mask,
            "loss_history": np.array(result.loss_history, dtype=object),
            "optic_second_order_fit": optic_second_order_fit,
            "optic_second_order_residual": optic_second_order_residual,
            "optic_second_order_fit_with_twist": optic_second_order_fit,
            "optic_second_order_residual_with_twist": optic_second_order_residual,
            "optic_second_order_fit_no_twist": optic_second_order_fit_no_twist,
            "optic_second_order_residual_no_twist": optic_second_order_residual_no_twist,
        })
        add_second_order_npz_outputs(
            output_arrays,
            "adp",
            result.optic,
            optic_second_order_fit,
            optic_second_order_residual,
            second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "autodiff",
            result.optic,
            optic_second_order_fit,
            optic_second_order_residual,
            second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "adp_with_twist",
            result.optic,
            optic_second_order_fit,
            optic_second_order_residual,
            second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "adp_no_twist",
            result.optic,
            optic_second_order_fit_no_twist,
            optic_second_order_residual_no_twist,
            second_order_summary_no_twist,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "autodiff_with_twist",
            result.optic,
            optic_second_order_fit,
            optic_second_order_residual,
            second_order_summary,
            pixel_spacing,
        )
        add_second_order_npz_outputs(
            output_arrays,
            "autodiff_no_twist",
            result.optic,
            optic_second_order_fit_no_twist,
            optic_second_order_residual_no_twist,
            second_order_summary_no_twist,
            pixel_spacing,
        )
        if result.summary.get("optic_model") != "pixel" and result.summary.get("optic_poly_coefficients") is not None:
            output_arrays["optic_poly_coefficients"] = np.asarray(result.summary["optic_poly_coefficients"], dtype=np.float64)
            output_arrays["optic_poly_terms"] = np.asarray(result.summary.get("optic_poly_terms", []), dtype=np.int64)
        summary.update(result.summary)
        summary["reconstruction_error"] = {
            "model": "processed input measurement - reconstructed measurement",
            "global_rms_nm": finite_rms(reconstruction_error),
            "global_pv_nm": finite_pv(reconstruction_error),
            "frame_rms_mean_nm": float(np.nanmean(reconstruction_error_rms_by_frame)),
            "frame_rms_max_nm": float(np.nanmax(reconstruction_error_rms_by_frame)),
        }
        summary["autodiff_second_order_fit"] = second_order_summary
        summary["autodiff_second_order_fit_with_twist"] = second_order_summary
        summary["autodiff_second_order_fit_no_twist"] = second_order_summary_no_twist
        summary["second_order_fit"] = second_order_summary
        summary["second_order_fit_with_twist"] = second_order_summary
        summary["second_order_fit_no_twist"] = second_order_summary_no_twist

    if args.mode == "both" and result is not None and sequential_optic is not None:
        raw_difference, raw_alignment = aligned_difference_remove_piston_tilt(result.optic, sequential_optic, pixel_spacing)
        save_tiff(output_dir / "autodiff_minus_sequential_raw_aligned.tiff", raw_difference)
        save_map_figure(
            output_dir / "autodiff_minus_sequential_raw_aligned.png",
            raw_difference,
            "ADP/autodiff optic minus sequential after piston/tilt alignment",
        )
        output_arrays["autodiff_minus_sequential_raw_aligned"] = raw_difference

        residual_difference = optic_second_order_residual - sequential_second_order_residual
        valid_residual_difference = np.isfinite(optic_second_order_residual) & np.isfinite(sequential_second_order_residual)
        residual_difference[~valid_residual_difference] = np.nan
        save_tiff(output_dir / "autodiff_minus_sequential.tiff", residual_difference)
        save_map_figure(
            output_dir / "autodiff_minus_sequential.png",
            residual_difference,
            "ADP residual minus sequential residual after independent 2nd-order fits, with twist",
        )
        save_tiff(output_dir / "autodiff_second_order_residual_minus_sequential.tiff", residual_difference)
        save_map_figure(
            output_dir / "autodiff_second_order_residual_minus_sequential.png",
            residual_difference,
            "ADP residual minus sequential residual after independent 2nd-order fits, with twist",
        )
        save_tiff(output_dir / "autodiff_minus_sequential_with_twist.tiff", residual_difference)
        save_map_figure(
            output_dir / "autodiff_minus_sequential_with_twist.png",
            residual_difference,
            "ADP residual minus sequential residual after independent 2nd-order fits, with twist",
        )
        save_tiff(output_dir / "autodiff_second_order_residual_minus_sequential_with_twist.tiff", residual_difference)

        residual_difference_no_twist = optic_second_order_residual_no_twist - sequential_second_order_residual_no_twist
        valid_residual_difference_no_twist = (
            np.isfinite(optic_second_order_residual_no_twist)
            & np.isfinite(sequential_second_order_residual_no_twist)
        )
        residual_difference_no_twist[~valid_residual_difference_no_twist] = np.nan
        save_tiff(output_dir / "autodiff_minus_sequential_no_twist.tiff", residual_difference_no_twist)
        save_map_figure(
            output_dir / "autodiff_minus_sequential_no_twist.png",
            residual_difference_no_twist,
            "ADP residual minus sequential residual after independent 2nd-order fits, no twist",
        )
        save_tiff(
            output_dir / "autodiff_second_order_residual_minus_sequential_no_twist.tiff",
            residual_difference_no_twist,
        )
        save_map_figure(
            output_dir / "autodiff_second_order_residual_minus_sequential_no_twist.png",
            residual_difference_no_twist,
            "ADP residual minus sequential residual after independent 2nd-order fits, no twist",
        )
        output_arrays["autodiff_minus_sequential"] = residual_difference
        output_arrays["autodiff_minus_sequential_with_twist"] = residual_difference
        output_arrays["autodiff_minus_sequential_no_twist"] = residual_difference_no_twist
        output_arrays["autodiff_second_order_residual_minus_sequential"] = residual_difference
        output_arrays["autodiff_second_order_residual_minus_sequential_with_twist"] = residual_difference
        output_arrays["autodiff_second_order_residual_minus_sequential_no_twist"] = residual_difference_no_twist
        comparison_with_twist = {
            "residual_difference_rms_nm": finite_rms(residual_difference),
            "residual_difference_pv_nm": finite_pv(residual_difference),
            "raw_aligned_difference_rms_nm": finite_rms(raw_difference),
            "raw_aligned_difference_pv_nm": finite_pv(raw_difference),
            "raw_alignment_removed": raw_alignment,
            "twist_mode": "with_twist",
            "interpretation": "autodiff_minus_sequential is the with-twist comparison for backward compatibility. It compares the two second-order residual maps after fitting qxy. The raw piston/tilt-aligned optic difference is saved separately as autodiff_minus_sequential_raw_aligned.",
        }
        comparison_no_twist = {
            "residual_difference_rms_nm": finite_rms(residual_difference_no_twist),
            "residual_difference_pv_nm": finite_pv(residual_difference_no_twist),
            "raw_aligned_difference_rms_nm": finite_rms(raw_difference),
            "raw_aligned_difference_pv_nm": finite_pv(raw_difference),
            "raw_alignment_removed": raw_alignment,
            "twist_mode": "no_twist",
            "interpretation": "No-twist comparison fits only c, x, y, x^2, and y^2. The qxy twist term is not subtracted, so twist remains in the residual for final analysis.",
        }
        summary["autodiff_vs_sequential"] = comparison_with_twist
        summary["autodiff_vs_sequential_with_twist"] = comparison_with_twist
        summary["autodiff_vs_sequential_no_twist"] = comparison_no_twist

    npz_path = output_dir / "stitching_result.npz"
    summary_path = output_dir / "summary.json"
    np.savez_compressed(npz_path, **output_arrays)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_parameters(
        run_parameters_path,
        args,
        {
            "status": "completed",
            "outputs": {
                "npz": str(npz_path.resolve()),
                "summary_json": str(summary_path.resolve()),
                "run_parameters_json": str(run_parameters_path.resolve()),
            },
            "summary_subset": {
                "final_rms": summary.get("final_rms"),
                "systematic_rms": summary.get("systematic_rms"),
                "sequential_radius_x_m": summary.get("sequential_second_order_fit", {}).get("radius_x_m") if isinstance(summary.get("sequential_second_order_fit"), dict) else None,
                "sequential_principal_radius_m": summary.get("sequential_second_order_fit", {}).get("principal_radius_m") if isinstance(summary.get("sequential_second_order_fit"), dict) else None,
                "sequential_qxy_nm_per_mm2": second_order_qxy(summary.get("sequential_second_order_fit_with_twist")),
                "sequential_principal_radius_no_twist_m": summary.get("sequential_second_order_fit_no_twist", {}).get("principal_radius_m") if isinstance(summary.get("sequential_second_order_fit_no_twist"), dict) else None,
                "adp_radius_x_m": summary.get("autodiff_second_order_fit", {}).get("radius_x_m") if isinstance(summary.get("autodiff_second_order_fit"), dict) else None,
                "adp_principal_radius_m": summary.get("autodiff_second_order_fit", {}).get("principal_radius_m") if isinstance(summary.get("autodiff_second_order_fit"), dict) else None,
                "adp_qxy_nm_per_mm2": second_order_qxy(summary.get("autodiff_second_order_fit_with_twist")),
                "adp_principal_radius_no_twist_m": summary.get("autodiff_second_order_fit_no_twist", {}).get("principal_radius_m") if isinstance(summary.get("autodiff_second_order_fit_no_twist"), dict) else None,
                "known_radius_constraint": summary.get("known_radius_constraint"),
            },
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir.resolve()}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch preprocessed interferometer data with sequential and polynomial ADP modes.")
    parser.add_argument("--input", default="processed_stitching_input.npz", help="Preprocessed NPZ from preprocess_interferometer_data.py.")
    parser.add_argument("--output-dir", default="stitching_outputs", help="Output directory for TIFF/PNG/NPZ/JSON results.")
    parser.add_argument("--mode", default="both", choices=["sequential", "autodiff", "both"], help="Calculation mode: sequential baseline, polynomial ADP, or both.")
    parser.add_argument("--adp-init", default="sequential", choices=["sequential", "warm_average"], help="Initial optic for ADP. Use sequential for 1-D curved mirrors.")
    parser.add_argument("--optic-model", default="poly", choices=["poly", "pixel"], help="Optic representation. poly is recommended; pixel is diagnostic only.")
    parser.add_argument("--optic-poly-order", type=int, default=10, help="Total 2D Legendre order when anisotropic orders are not supplied.")
    parser.add_argument("--optic-poly-order-x", type=int, default=None, help="Optional Legendre order along output X/scan direction.")
    parser.add_argument("--optic-poly-order-y", type=int, default=None, help="Optional Legendre order along output Y/cross-scan direction.")
    parser.add_argument("--optic-poly-basis", default="legendre", choices=["legendre"], help="Polynomial basis family; only Legendre is implemented.")
    parser.add_argument("--adp-curvature-anchor", default="initial", choices=["initial", "none"], help="Pixel-mode curvature anchor; inactive for polynomial mode.")
    parser.add_argument("--adp-prior-l2", type=float, default=0.005, help="Pixel/residual-image prior toward the initial optic; mainly diagnostic for pixel mode.")
    parser.add_argument("--curv-anchor-l2", type=float, default=100.0, help="Pixel-mode soft curvature-anchor weight; inactive for polynomial mode.")
    parser.add_argument("--known-radius-m", type=float, default=None, help="Optional known mirror radius in meters. Adds a soft ADP constraint on q=500/R in nm/mm^2; default disables the constraint.")
    parser.add_argument("--known-radius-axis", default="x", choices=["x", "y", "principal"], help="Axis used by --known-radius-m: stitched x curvature, stitched y curvature, or signed principal curvature.")
    parser.add_argument("--known-radius-l2", type=float, default=100.0, help="Weight for the known-radius soft constraint. Active only when --known-radius-m is supplied.")
    parser.add_argument("--optic-coeff-l2", type=float, default=0.0, help="Degree-weighted L2 regularization for polynomial optic coefficients.")
    parser.add_argument("--optic-coeff-degree-power", type=float, default=4.0, help="Polynomial coefficient regularization degree exponent.")
    parser.add_argument("--lam-g", type=float, default=1e-6, help="Universal gauge penalty for optic piston and plane means.")
    parser.add_argument("--accept-tol", type=float, default=1.05, help="Accept candidate state when total loss is within this factor of best loss.")
    parser.add_argument("--rollback-factor", type=float, default=1.5, help="Rollback to best state when total loss exceeds this factor of best loss.")
    parser.add_argument("--s-mode", default="fixed_from_residual", choices=["fixed_from_residual", "none"], help="Use frozen high-pass detector residual systematic, or disable blind systematic handling.")
    parser.add_argument("--s-highpass-order", type=int, default=3, help="Detector Legendre order removed from residual-estimated systematic.")
    parser.add_argument("--fit-positions", action="store_true", help="Refine small residual scan-position corrections during ADP.")
    parser.add_argument("--plane-order", default="tilt", choices=["piston", "tilt"], help="Per-frame nuisance model: piston only, or piston plus x/y tilt.")
    parser.add_argument("--interp-mode", default="bilinear", choices=["bilinear", "bicubic"], help="Interpolation used to sample the global optic into each aperture.")
    parser.add_argument("--min-output-coverage", type=float, default=0.15, help="Hide output optic pixels below this normalized coverage.")
    parser.add_argument("--smoothness", type=float, default=0.0, help="Second-difference smoothness on optic image; mainly for pixel diagnostic mode.")
    parser.add_argument("--seam-smoothness", type=float, default=0.0, help="Coverage-edge smoothness on optic image; mainly for pixel diagnostic mode.")
    parser.add_argument("--plane-l2", type=float, default=0.0, help="L2 penalty on per-frame tilt coefficients; piston is not penalized.")
    parser.add_argument("--position-l2", type=float, default=1.0, help="L2 penalty on refined position corrections in pixels.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of frames per loss batch.")
    parser.add_argument("--phase2-iters", type=int, default=300, help="Main Adam iterations for ADP refinement.")
    parser.add_argument("--lr-o", type=float, default=2e-2, help="Adam learning rate for optic parameters.")
    parser.add_argument("--lr-planes", type=float, default=1e-2, help="Adam learning rate for frame piston/tilt parameters.")
    parser.add_argument("--lr-pos", type=float, default=1e-3, help="Adam learning rate for position corrections.")
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda, or auto.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
