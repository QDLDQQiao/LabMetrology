#!/usr/bin/env python3
"""Autodiff interferometer systematic calibration from rotated optics data.

The forward model is

    M_i(x, y) = S(x, y) + rotate(O, theta_i, center)(x, y) + P_i(x, y)

where ``S`` is the fixed interferometer systematic error, ``O`` is the surface
fixed to the under-test optic, and ``P_i`` is a per-frame piston/tilt plane.
The solver follows ``plan.md``: PyTorch float64 tensors, ``grid_sample`` with
``align_corners=True``, learned center of rotation, and null-space projection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import translate_py as matio

torch.set_default_dtype(torch.float64)


@dataclass
class CalibrationInput:
    filename: str
    names: list[str]
    angles_deg: np.ndarray
    measurements: np.ndarray
    masks: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class CalibrationResult:
    systematic: np.ndarray
    optic: np.ndarray
    systematic_mask: np.ndarray
    optic_mask: np.ndarray
    planes: np.ndarray
    residuals: np.ndarray
    rms_history: list[float]
    angles_deg: np.ndarray
    names: list[str]
    bbox: tuple[int, int, int, int]
    rotation_center: tuple[float, float]
    loss_history: list[dict[str, Any]]


def parse_angle_deg(name: str) -> float:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:°|deg|degree)", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse rotation angle from measurement name: {name!r}")
    token = match.group(1)
    value = float(token)
    if token.startswith("-") and match.start(1) > 0:
        previous = name[match.start(1) - 1]
        if not previous.isspace() and previous not in "([{,:;":
            value = abs(value)
    return 0.0 if abs(value) < 1e-12 else value


def load_rotation_set(base_dir: Path, filename: str) -> CalibrationInput:
    phases, names = matio.load_phase_data(base_dir, filename)
    if len(phases) < 2:
        raise ValueError(f"{filename} must contain at least two rotation measurements")

    angles = np.array([parse_angle_deg(name) for name in names], dtype=np.float64)
    order = np.argsort(angles)
    phases = [phases[index] for index in order]
    names = [names[index] for index in order]
    angles = angles[order]

    raw = np.stack([np.asarray(phase, dtype=np.float64) for phase in phases])
    finite = np.isfinite(raw)
    aperture = np.any(finite, axis=0)
    rows = np.where(np.any(aperture, axis=1))[0]
    cols = np.where(np.any(aperture, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError(f"{filename} does not contain finite aperture samples")

    row0, row1 = int(rows[0]), int(rows[-1]) + 1
    col0, col1 = int(cols[0]), int(cols[-1]) + 1
    cropped = raw[:, row0:row1, col0:col1]
    masks = np.isfinite(cropped)
    measurements = cropped.copy()
    measurements[~masks] = np.nan
    return CalibrationInput(
        filename=filename,
        names=names,
        angles_deg=angles,
        measurements=measurements,
        masks=masks.astype(np.float64),
        bbox=(row0, row1, col0, col1),
    )


def downsample_input(data: CalibrationInput, factor: int) -> CalibrationInput:
    if factor <= 1:
        return data
    return CalibrationInput(
        filename=data.filename,
        names=data.names,
        angles_deg=data.angles_deg,
        measurements=data.measurements[:, ::factor, ::factor],
        masks=data.masks[:, ::factor, ::factor],
        bbox=data.bbox,
    )


def default_center(shape: tuple[int, int]) -> tuple[float, float]:
    rows, cols = shape
    return (rows - 1) / 2.0, (cols - 1) / 2.0


def finite_rms(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(values[finite] ** 2)))


def erode_mask(mask: torch.Tensor) -> torch.Tensor:
    eroded = -F.max_pool2d((-mask)[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    return (eroded > 0.5).to(mask.dtype)


class InterferometerModel(nn.Module):
    def __init__(
        self,
        measurements: np.ndarray,
        masks: np.ndarray,
        angles_deg: np.ndarray,
        init_center_yx: tuple[float, float],
        optic_mask: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        device = torch.device(device)
        measurements = np.asarray(measurements, dtype=np.float64)
        masks = np.asarray(masks, dtype=np.float64)
        finite = np.isfinite(measurements)
        clean_measurements = np.where(finite, measurements, 0.0)
        clean_masks = np.where(finite, masks, 0.0)

        m_tensor = torch.as_tensor(clean_measurements, dtype=torch.float64, device=device)
        mask_tensor = torch.as_tensor(clean_masks, dtype=torch.float64, device=device)
        angles = torch.as_tensor(np.deg2rad(angles_deg), dtype=torch.float64, device=device)
        self.register_buffer("M", m_tensor)
        self.register_buffer("mask", mask_tensor)
        self.register_buffer("angles", angles)

        n_frames, height, width = clean_measurements.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=torch.float64, device=device),
            torch.linspace(-1.0, 1.0, width, dtype=torch.float64, device=device),
            indexing="ij",
        )
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1))

        union_mask = (mask_tensor.sum(dim=0) > 0).to(torch.float64)
        self.register_buffer("union_mask", union_mask)

        if optic_mask is None:
            center_norm = self.pixel_center_to_norm(init_center_yx, height, width, device)
            mask_intersection = torch.ones((height, width), dtype=torch.float64, device=device)
            with torch.no_grad():
                for index in range(n_frames):
                    back = self.sample_image(
                        mask_tensor[index],
                        -angles[index : index + 1],
                        center_norm,
                        mode="nearest",
                    )[0]
                    mask_intersection = mask_intersection * (back > 0.5).to(torch.float64)
                optic_mask_tensor = erode_mask(mask_intersection)
        else:
            optic_mask_tensor = torch.as_tensor(optic_mask, dtype=torch.float64, device=device)
        self.register_buffer("optic_mask", optic_mask_tensor)

        tilt_x = xx.clone()
        tilt_y = yy.clone()
        denom = union_mask.sum().clamp_min(1.0)
        tilt_x = tilt_x - (tilt_x * union_mask).sum() / denom
        tilt_y = tilt_y - (tilt_y * union_mask).sum() / denom
        self.register_buffer("tilt_x", tilt_x)
        self.register_buffer("tilt_y", tilt_y)

        self.S = nn.Parameter(torch.zeros((height, width), dtype=torch.float64, device=device))
        self.O = nn.Parameter(torch.zeros((height, width), dtype=torch.float64, device=device))
        self.planes = nn.Parameter(torch.zeros((n_frames, 3), dtype=torch.float64, device=device))
        self.center_norm = nn.Parameter(self.pixel_center_to_norm(init_center_yx, height, width, device))

    @staticmethod
    def pixel_center_to_norm(
        center_yx: tuple[float, float],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        cy_pix, cx_pix = center_yx
        cx_norm = 2.0 * float(cx_pix) / (width - 1) - 1.0
        cy_norm = 2.0 * float(cy_pix) / (height - 1) - 1.0
        return torch.tensor([cx_norm, cy_norm], dtype=torch.float64, device=device)

    def norm_center_to_pixel(self) -> tuple[float, float]:
        height, width = self.S.shape
        center = self.center_norm.detach().cpu().numpy()
        cx_pix = (center[0] + 1.0) * (width - 1) / 2.0
        cy_pix = (center[1] + 1.0) * (height - 1) / 2.0
        return float(cy_pix), float(cx_pix)

    def rotation_grid(self, angles: torch.Tensor, center_norm: torch.Tensor | None = None) -> torch.Tensor:
        center = self.center_norm if center_norm is None else center_norm
        shifted = self.base_grid - center
        c = torch.cos(-angles)
        s = torch.sin(-angles)
        row0 = torch.stack([c, -s], dim=-1)
        row1 = torch.stack([s, c], dim=-1)
        rotation = torch.stack([row0, row1], dim=1)
        rotated = torch.einsum("nij,hwj->nhwi", rotation, shifted)
        return rotated + center

    def sample_image(
        self,
        image: torch.Tensor,
        angles: torch.Tensor,
        center_norm: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        grid = self.rotation_grid(angles, center_norm=center_norm)
        image_in = image[None, None].expand(len(angles), -1, -1, -1)
        return F.grid_sample(
            image_in,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    def get_rotated_optic(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_grid = self.rotation_grid(self.angles)
        optic_in = (self.O * self.optic_mask)[None, None].expand(len(self.angles), -1, -1, -1)
        optic_rot = F.grid_sample(
            optic_in,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        inside = (sample_grid.abs() <= 1.0).all(dim=-1).to(torch.float64)
        mask_in = self.optic_mask[None, None].expand(len(self.angles), -1, -1, -1)
        optic_mask_rot = F.grid_sample(
            mask_in,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        valid_rot = inside * (optic_mask_rot > 0.5).to(torch.float64)
        return optic_rot, valid_rot

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        optic_rot, valid_rot = self.get_rotated_optic()
        planes = (
            self.planes[:, 0, None, None]
            + self.planes[:, 1, None, None] * self.tilt_x[None]
            + self.planes[:, 2, None, None] * self.tilt_y[None]
        )
        prediction = self.S[None] + optic_rot + planes
        total_mask = self.mask * valid_rot
        return prediction, total_mask


def plane_moments_loss(surface: torch.Tensor, mask: torch.Tensor, tilt_x: torch.Tensor, tilt_y: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return surface.sum() * 0.0
    piston = surface[valid].mean() ** 2
    tx_denom = ((tilt_x**2) * mask).sum().clamp_min(1.0)
    ty_denom = ((tilt_y**2) * mask).sum().clamp_min(1.0)
    tilt_x_loss = ((surface * tilt_x * mask).sum() / tx_denom) ** 2
    tilt_y_loss = ((surface * tilt_y * mask).sum() / ty_denom) ** 2
    return piston + tilt_x_loss + tilt_y_loss


def compute_loss(model: InterferometerModel, reg_lambda: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction, total_mask = model()
    residual = (model.M - prediction) * total_mask
    n_valid = total_mask.sum().clamp_min(1.0)
    loss_data = (residual**2).sum() / n_valid
    loss_null = plane_moments_loss(model.O, model.optic_mask, model.tilt_x, model.tilt_y)
    loss_null = loss_null + plane_moments_loss(model.S, model.union_mask, model.tilt_x, model.tilt_y)
    return loss_data + reg_lambda * loss_null, loss_data, total_mask


def fit_plane_torch(data: torch.Tensor, mask: torch.Tensor, tilt_x: torch.Tensor, tilt_y: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if valid.sum() < 3:
        return torch.zeros(3, dtype=data.dtype, device=data.device)
    design = torch.stack([torch.ones_like(tilt_x)[valid], tilt_x[valid], tilt_y[valid]], dim=1)
    target = data[valid]
    return torch.linalg.lstsq(design, target).solution


def subtract_plane_inplace(
    surface: torch.Tensor,
    mask: torch.Tensor,
    tilt_x: torch.Tensor,
    tilt_y: torch.Tensor,
) -> torch.Tensor:
    coeff = fit_plane_torch(surface, mask, tilt_x, tilt_y)
    surface -= coeff[0] + coeff[1] * tilt_x + coeff[2] * tilt_y
    return coeff


def postprocess_null_space(model: InterferometerModel, project_rotational_mean: bool) -> None:
    with torch.no_grad():
        valid_o = model.optic_mask > 0.5
        if torch.any(valid_o):
            piston_o = model.O[valid_o].mean()
            model.O -= piston_o
            model.planes[:, 0] += piston_o

            tilt_coeff = fit_plane_torch(model.O, model.optic_mask, model.tilt_x, model.tilt_y)
            removed = tilt_coeff[1] * model.tilt_x + tilt_coeff[2] * model.tilt_y
            model.O -= removed
            removed_rot = model.sample_image(removed * model.optic_mask, model.angles, model.center_norm)
            for index in range(len(model.angles)):
                coeff = fit_plane_torch(removed_rot[index], model.mask[index], model.tilt_x, model.tilt_y)
                model.planes[index] += coeff

        if project_rotational_mean:
            optic_rot, valid_rot = model.get_rotated_optic()
            weights = model.mask * valid_rot
            denom = weights.sum(dim=0).clamp_min(1.0)
            optic_sym = (optic_rot * weights).sum(dim=0) / denom
            optic_sym = optic_sym * (weights.sum(dim=0) > 0).to(torch.float64)
            model.S += optic_sym
            model.O -= optic_sym * model.optic_mask

        coeff_s = subtract_plane_inplace(model.S, model.union_mask, model.tilt_x, model.tilt_y)
        model.planes += coeff_s[None]


def residual_numpy(model: InterferometerModel) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        prediction, total_mask = model()
        residual = (model.M - prediction) * total_mask
        valid = total_mask > 0.5
        rms_value = torch.sqrt((residual[valid] ** 2).mean()).detach().cpu().item() if torch.any(valid) else float("nan")
        residual_np = residual.detach().cpu().numpy().astype(np.float64)
        mask_np = valid.detach().cpu().numpy()
        residual_np[~mask_np] = np.nan
    return residual_np, float(rms_value)


def solve_calibration_autodiff(
    data: CalibrationInput,
    init_center_yx: tuple[float, float] | None = None,
    phase_a_iters: int = 300,
    phase_b_iters: int = 1000,
    lbfgs_iters: int = 0,
    lr_surface: float = 1e-1,
    lr_planes: float = 1e-2,
    lr_center: float = 1e-3,
    reg_lambda: float | None = None,
    project_rotational_mean: bool = False,
    device: str = "cpu",
    log_every: int = 50,
    tolerance: float = 1e-7,
) -> CalibrationResult:
    measurements = data.measurements
    masks = data.masks
    shape = measurements.shape[1:]
    init_center_yx = default_center(shape) if init_center_yx is None else init_center_yx
    model = InterferometerModel(measurements, masks, data.angles_deg, init_center_yx, device=device)

    finite_values = measurements[np.isfinite(measurements)]
    data_scale = float(np.sqrt(np.mean(finite_values**2))) if finite_values.size else 1.0
    if reg_lambda is None:
        reg_lambda = 1e-6 * max(data_scale**2, 1.0)

    rms_history: list[float] = []
    loss_history: list[dict[str, Any]] = []
    global_step = 0

    def log_state(label: str, step: int, steps: int, loss: torch.Tensor, loss_data: torch.Tensor) -> None:
        nonlocal global_step
        rms_value = torch.sqrt(loss_data.detach()).cpu().item()
        center_y, center_x = model.norm_center_to_pixel()
        loss_record = {
            "global_step": global_step,
            "phase": label,
            "phase_step": step,
            "phase_steps": steps,
            "loss": float(loss.detach().cpu().item()),
            "loss_data": float(loss_data.detach().cpu().item()),
            "rms": float(rms_value),
            "center_y": float(center_y),
            "center_x": float(center_x),
        }
        rms_history.append(float(rms_value))
        loss_history.append(loss_record)
        print(f"    {label} {step:5d}/{steps}: RMS={rms_value:.6g}")

    def run_adam(parameters: list[dict[str, Any]], steps: int, label: str) -> None:
        nonlocal global_step
        if steps <= 0:
            return
        optimizer = torch.optim.Adam(parameters)
        recent: list[float] = []
        for step in range(steps):
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            loss, loss_data, _ = compute_loss(model, reg_lambda)
            loss.backward()
            optimizer.step()
            if step % log_every == 0 or step == steps - 1:
                log_state(label, step + 1, steps, loss, loss_data)
            loss_value = loss_data.detach().cpu().item()
            recent.append(loss_value)
            if len(recent) > 50:
                previous = recent.pop(0)
                rel = abs(previous - loss_value) / max(abs(previous), 1.0)
                if rel < tolerance:
                    log_state(label, step + 1, steps, loss, loss_data)
                    print(f"    {label} stopped at {step + 1}")
                    break

    model.center_norm.requires_grad_(False)
    run_adam(
        [
            {"params": [model.S, model.O], "lr": lr_surface},
            {"params": [model.planes], "lr": lr_planes},
        ],
        phase_a_iters,
        "phase A",
    )

    model.center_norm.requires_grad_(True)
    run_adam(
        [
            {"params": [model.S, model.O], "lr": lr_surface * 0.3},
            {"params": [model.planes], "lr": lr_planes},
            {"params": [model.center_norm], "lr": lr_center},
        ],
        phase_b_iters,
        "phase B",
    )

    if lbfgs_iters > 0:
        model.center_norm.requires_grad_(False)
        optimizer = torch.optim.LBFGS([model.S, model.O, model.planes], max_iter=lbfgs_iters, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = compute_loss(model, reg_lambda)
            loss.backward()
            return loss

        optimizer.step(closure)
        _, rms_after = residual_numpy(model)
        center_y, center_x = model.norm_center_to_pixel()
        rms_history.append(rms_after)
        loss_history.append(
            {
                "global_step": global_step,
                "phase": "L-BFGS",
                "phase_step": lbfgs_iters,
                "phase_steps": lbfgs_iters,
                "loss": float(rms_after**2),
                "loss_data": float(rms_after**2),
                "rms": float(rms_after),
                "center_y": float(center_y),
                "center_x": float(center_x),
            }
        )
        print(f"    L-BFGS polish: RMS={rms_after:.6g}")

    postprocess_null_space(model, project_rotational_mean=project_rotational_mean)
    residuals, final_rms = residual_numpy(model)
    rms_history.append(final_rms)
    center_y, center_x = model.norm_center_to_pixel()
    loss_history.append(
        {
            "global_step": global_step,
            "phase": "postprocess",
            "phase_step": 0,
            "phase_steps": 0,
            "loss": float(final_rms**2),
            "loss_data": float(final_rms**2),
            "rms": float(final_rms),
            "center_y": float(center_y),
            "center_x": float(center_x),
        }
    )

    with torch.no_grad():
        systematic = model.S.detach().cpu().numpy().astype(np.float64)
        optic = model.O.detach().cpu().numpy().astype(np.float64)
        systematic_mask = model.union_mask.detach().cpu().numpy().astype(np.float64)
        optic_mask = model.optic_mask.detach().cpu().numpy().astype(np.float64)
        planes = model.planes.detach().cpu().numpy().astype(np.float64)
    systematic[systematic_mask <= 0.5] = np.nan
    optic[optic_mask <= 0.5] = np.nan

    return CalibrationResult(
        systematic=systematic,
        optic=optic,
        systematic_mask=systematic_mask,
        optic_mask=optic_mask,
        planes=planes,
        residuals=residuals,
        rms_history=rms_history,
        angles_deg=data.angles_deg,
        names=data.names,
        bbox=data.bbox,
        rotation_center=model.norm_center_to_pixel(),
        loss_history=loss_history,
    )


def solve_calibration(
    data: CalibrationInput,
    iterations: int = 1000,
    fit_planes: bool = True,
    project_rotational_mean: bool = False,
    rotation_center: tuple[float, float] | None = None,
    optic_refine_steps: int = 0,
    optic_refine_gain: float = 0.0,
    max_optic_rms: float | None = None,
    phase_a_iters: int = 300,
    lbfgs_iters: int = 0,
    device: str = "cpu",
) -> CalibrationResult:
    if not fit_planes:
        print("Warning: autodiff model always includes plane parameters; use post-analysis to ignore them if needed.")
    _ = optic_refine_steps, optic_refine_gain, max_optic_rms
    return solve_calibration_autodiff(
        data,
        init_center_yx=rotation_center,
        phase_a_iters=phase_a_iters,
        phase_b_iters=iterations,
        lbfgs_iters=lbfgs_iters,
        project_rotational_mean=project_rotational_mean,
        device=device,
    )


def save_image_grid(
    maps: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    cmap: str = "viridis",
    symmetric: bool = True,
    ncols: int = 3,
    shared_scale: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Skipping {output_path.name}: matplotlib is not installed in this Python environment.")
        return

    nrows = int(np.ceil(len(maps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.1 * nrows), squeeze=False, constrained_layout=True)
    finite_lists = [data[np.isfinite(data)].ravel() for data in maps if np.isfinite(data).any()]
    finite_values = np.concatenate(finite_lists) if finite_lists else np.array([], dtype=float)
    shared_limits = (None, None)
    if shared_scale and finite_values.size:
        if symmetric:
            limit = float(np.nanpercentile(np.abs(finite_values), 99))
            shared_limits = (-limit, limit)
        else:
            shared_limits = tuple(np.nanpercentile(finite_values, [1, 99]))

    for ax in axes.ravel():
        ax.axis("off")
    for ax, data, title in zip(axes.ravel(), maps, titles):
        vmin, vmax = shared_limits
        finite = data[np.isfinite(data)]
        if not shared_scale and finite.size:
            if symmetric:
                limit = float(np.nanpercentile(np.abs(finite), 99))
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = np.nanpercentile(finite, [1, 99])
        image = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.axis("on")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_history_outputs(result: CalibrationResult, output_dir: Path, stem: str) -> None:
    history_path = output_dir / f"{stem}_optimization_history.json"
    history_path.write_text(json.dumps(result.loss_history, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{stem}_optimization_history.csv"
    fields = ["global_step", "phase", "phase_step", "phase_steps", "loss", "loss_data", "rms", "center_y", "center_x"]
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for record in result.loss_history:
            handle.write(",".join(str(record.get(field, "")) for field in fields) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Skipping history plots for {stem}: matplotlib is not installed in this Python environment.")
        return

    if not result.loss_history:
        return

    steps = np.array([record["global_step"] for record in result.loss_history], dtype=float)
    loss = np.array([record["loss"] for record in result.loss_history], dtype=float)
    loss_data = np.array([record["loss_data"] for record in result.loss_history], dtype=float)
    rms_values = np.array([record["rms"] for record in result.loss_history], dtype=float)
    center_y = np.array([record["center_y"] for record in result.loss_history], dtype=float)
    center_x = np.array([record["center_x"] for record in result.loss_history], dtype=float)
    phases = [record["phase"] for record in result.loss_history]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True, sharex=True)
    axes[0].semilogy(steps, loss, marker="o", label="total loss")
    axes[0].semilogy(steps, loss_data, marker="s", label="data loss")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{stem} loss history")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, rms_values, marker="o", color="tab:blue")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("RMS residual")
    axes[1].grid(True, alpha=0.3)
    for step, phase in zip(steps, phases):
        if phase in {"phase A", "phase B", "L-BFGS", "postprocess"}:
            axes[1].annotate(phase, (step, rms_values[np.where(steps == step)[0][0]]), fontsize=7, alpha=0.7)
    fig.savefig(output_dir / f"{stem}_loss_vs_iteration.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), constrained_layout=True, sharex=True)
    axes[0].plot(steps, center_y, marker="o", label="center y")
    axes[0].plot(steps, center_x, marker="s", label="center x")
    axes[0].set_ylabel("center (cropped pixels)")
    axes[0].set_title(f"{stem} rotation center history")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, center_y - center_y[0], marker="o", label="delta y")
    axes[1].plot(steps, center_x - center_x[0], marker="s", label="delta x")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("offset from initial (pixels)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(output_dir / f"{stem}_center_vs_iteration.png", dpi=160)
    plt.close(fig)


def save_plane_plot(result: CalibrationResult, output_dir: Path, stem: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = np.arange(len(result.angles_deg))
    labels = [f"{angle:g} deg" for angle in result.angles_deg]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)
    names = ["piston", "tilt x", "tilt y"]
    for index, ax in enumerate(axes):
        ax.plot(x, result.planes[:, index], marker="o")
        ax.set_ylabel(names[index])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_title(f"{stem} per-frame plane coefficients")
    fig.savefig(output_dir / f"{stem}_planes_by_angle.png", dpi=160)
    plt.close(fig)


def save_outputs(data: CalibrationInput, result: CalibrationResult, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(data.filename).stem

    save_image_grid(
        [result.systematic, result.optic],
        ["fixed systematic S", "optic-fixed term O"],
        output_dir / f"{stem}_calibrated_maps.png",
        ncols=2,
        shared_scale=False,
    )
    save_image_grid(
        [result.residuals[index] for index in range(result.residuals.shape[0])],
        [f"residual {angle:g} deg" for angle in result.angles_deg],
        output_dir / f"{stem}_residuals.png",
    )
    save_image_grid(
        [data.measurements[index] for index in range(data.measurements.shape[0])],
        [f"measurement {angle:g} deg" for angle in result.angles_deg],
        output_dir / f"{stem}_input_measurements.png",
    )
    save_image_grid(
        [result.systematic],
        ["reconstructed fixed systematic error S"],
        output_dir / f"{stem}_systematic_error.png",
        ncols=1,
        shared_scale=False,
    )
    save_image_grid(
        [result.optic],
        ["reconstructed optic-fixed error O"],
        output_dir / f"{stem}_optic_error.png",
        ncols=1,
        shared_scale=False,
    )

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.plot(np.arange(1, len(result.rms_history) + 1), result.rms_history, marker="o")
        ax.set_xlabel("logged optimization step")
        ax.set_ylabel("RMS residual")
        ax.set_title(f"{stem} convergence")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / f"{stem}_convergence.png", dpi=160)
        plt.close(fig)
    else:
        print(f"Skipping {stem}_convergence.png: matplotlib is not installed in this Python environment.")

    save_history_outputs(result, output_dir, stem)
    save_plane_plot(result, output_dir, stem)

    np.savez_compressed(
        output_dir / f"{stem}_calibration.npz",
        systematic=result.systematic,
        optic=result.optic,
        systematic_mask=result.systematic_mask,
        optic_mask=result.optic_mask,
        residuals=result.residuals,
        planes=result.planes,
        angles_deg=result.angles_deg,
        rms_history=np.array(result.rms_history),
        bbox=np.array(result.bbox),
        rotation_center=np.array(result.rotation_center),
        loss_history=np.array(result.loss_history, dtype=object),
        names=np.array(result.names, dtype=object),
    )

    summary: dict[str, Any] = {
        "filename": data.filename,
        "names": result.names,
        "angles_deg": result.angles_deg.tolist(),
        "bbox_row0_row1_col0_col1": list(result.bbox),
        "logged_iterations": len(result.rms_history),
        "initial_logged_rms": result.rms_history[0] if result.rms_history else None,
        "final_rms": result.rms_history[-1] if result.rms_history else None,
        "input_rms": finite_rms(data.measurements),
        "systematic_rms": finite_rms(result.systematic),
        "optic_rms": finite_rms(result.optic),
        "rotation_center_yx_in_cropped_pixels": list(result.rotation_center),
        "rotation_center_yx_in_original_pixels": [
            result.rotation_center[0] + result.bbox[0],
            result.rotation_center[1] + result.bbox[2],
        ],
        "plane_coefficients_piston_x_y": result.planes.tolist(),
        "optimization_history_json": f"{stem}_optimization_history.json",
        "optimization_history_csv": f"{stem}_optimization_history.csv",
        "output_npz": f"{stem}_calibration.npz",
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def synthetic_test(device: str) -> None:
    rng = np.random.default_rng(3)
    height = width = 64
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    mask = ((xx**2 + yy**2) <= 0.82**2).astype(np.float64)
    systematic = 0.8 * xx + 0.5 * yy + 0.4 * np.sin(2 * np.pi * xx)
    optic = 0.6 * np.sin(3 * np.pi * (xx + 0.2 * yy)) + 0.2 * rng.normal(size=(height, width))
    systematic *= mask
    optic *= mask
    angles = np.arange(0, 360, 45, dtype=np.float64)
    center = ((height - 1) / 2 + 1.7, (width - 1) / 2 - 2.3)

    tmp = CalibrationInput(
        filename="synthetic",
        names=[f"{a:g} deg" for a in angles],
        angles_deg=angles,
        measurements=np.zeros((len(angles), height, width), dtype=np.float64),
        masks=np.broadcast_to(mask, (len(angles), height, width)).copy(),
        bbox=(0, height, 0, width),
    )
    synth_model = InterferometerModel(tmp.measurements, tmp.masks, angles, center, optic_mask=mask, device=device)
    with torch.no_grad():
        synth_model.S.copy_(torch.as_tensor(systematic, dtype=torch.float64, device=device))
        synth_model.O.copy_(torch.as_tensor(optic, dtype=torch.float64, device=device))
        synth_model.center_norm.copy_(synth_model.pixel_center_to_norm(center, height, width, torch.device(device)))
        synth_model.planes.copy_(torch.as_tensor(rng.normal(scale=0.05, size=(len(angles), 3)), dtype=torch.float64, device=device))
        measurements, total_mask = synth_model()
    measured = measurements.detach().cpu().numpy()
    measured += rng.normal(scale=0.01, size=measured.shape)
    measured[total_mask.detach().cpu().numpy() <= 0.5] = np.nan
    test_data = CalibrationInput("synthetic", tmp.names, angles, measured, np.isfinite(measured).astype(float), tmp.bbox)
    result = solve_calibration_autodiff(
        test_data,
        init_center_yx=default_center((height, width)),
        phase_a_iters=80,
        phase_b_iters=160,
        lr_surface=5e-2,
        lr_planes=1e-2,
        lr_center=5e-3,
        project_rotational_mean=False,
        device=device,
        log_every=80,
    )
    center_err = np.hypot(result.rotation_center[0] - center[0], result.rotation_center[1] - center[1])
    print(f"Synthetic final RMS: {result.rms_history[-1]:.6g}; center error: {center_err:.3f} px")
    if not np.isfinite(result.rms_history[-1]) or result.rms_history[-1] > 0.15:
        raise RuntimeError("Synthetic calibration residual is too high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="circle_ref", help="Directory containing circ_rot_*.mat files.")
    parser.add_argument("--files", nargs="+", default=["circ_rot_1.mat", "circ_rot_2.mat"])
    parser.add_argument("--output-dir", default="calibration_outputs_autodiff")
    parser.add_argument("--phase-a-iters", type=int, default=300)
    parser.add_argument("--phase-b-iters", type=int, default=1000)
    parser.add_argument("--lbfgs-iters", type=int, default=0)
    parser.add_argument("--lr-surface", type=float, default=1e-1)
    parser.add_argument("--lr-planes", type=float, default=1e-2)
    parser.add_argument("--lr-center", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50, help="Record optimizer diagnostics every N steps.")
    parser.add_argument("--downsample", type=int, default=1, help="Optional stride downsampling for quick experiments.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    parser.add_argument("--project-rotational-mean", action="store_true")
    parser.add_argument("--synthetic-test", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def main() -> None:
    args = parse_args()
    if not args.show:
        try:
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            pass
    device = resolve_device(args.device)
    print(f"Using torch {torch.__version__} on {device} with dtype float64")

    if args.synthetic_test:
        synthetic_test(device)
        return

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in args.files:
        print(f"Calibrating {filename} ...")
        data = load_rotation_set(Path(args.base_dir), filename)
        if args.downsample > 1:
            data = downsample_input(data, args.downsample)
            print(f"  downsampled by {args.downsample}; shape={data.measurements.shape[1:]}")

        result = solve_calibration_autodiff(
            data,
            init_center_yx=default_center(data.measurements.shape[1:]),
            phase_a_iters=args.phase_a_iters,
            phase_b_iters=args.phase_b_iters,
            lbfgs_iters=args.lbfgs_iters,
            lr_surface=args.lr_surface,
            lr_planes=args.lr_planes,
            lr_center=args.lr_center,
            reg_lambda=args.reg_lambda,
            project_rotational_mean=args.project_rotational_mean,
            device=device,
            log_every=args.log_every,
        )
        save_outputs(data, result, output_root / Path(filename).stem)
        print(
            f"  angles: {', '.join(f'{angle:g} deg' for angle in result.angles_deg)} | "
            f"final RMS: {result.rms_history[-1]:.6g} | "
            f"center y,x: {result.rotation_center[0]:.3f}, {result.rotation_center[1]:.3f} | "
            f"S RMS: {finite_rms(result.systematic):.6g} | O RMS: {finite_rms(result.optic):.6g}"
        )
    print(f"Calibration outputs written to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
