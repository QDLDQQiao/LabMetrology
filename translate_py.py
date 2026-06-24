#!/usr/bin/env python3
"""Process Zygo-like interferometer mirror metrology data.

This file is a runnable Python version of the exploratory MATLAB workflow in
``circle_ref/jtec_holder_measurements.m``.  It loads ``adata.Phase`` and
``adata.Name`` from MATLAB v5 ``.mat`` files, crops invalid borders, plots the
sub-apertures, demonstrates quadrant resampling, and performs a low-order
Legendre fit on the reference circular measurement.
"""

from __future__ import annotations

import argparse
import os
import struct
import zlib
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

try:
    import scipy.io as sio
except ImportError:  # The fallback reader below covers the files in circle_ref.
    sio = None


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15
MI_UTF8 = 16
MI_UTF16 = 17
MI_UTF32 = 18

MX_CELL = 1
MX_STRUCT = 2
MX_CHAR = 4
MX_DOUBLE = 6

MAT_DTYPE = {
    MI_INT8: "i1",
    MI_UINT8: "u1",
    MI_INT16: "<i2",
    MI_UINT16: "<u2",
    MI_INT32: "<i4",
    MI_UINT32: "<u4",
    MI_SINGLE: "<f4",
    MI_DOUBLE: "<f8",
    MI_INT64: "<i8",
    MI_UINT64: "<u8",
}


class MatReader:
    """Small MATLAB v5 element reader for numeric/cell/struct/char data."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def element(self) -> tuple[int, bytes] | None:
        if self.offset >= len(self.data):
            return None

        tag = struct.unpack_from("<I", self.data, self.offset)[0]
        small_nbytes = tag >> 16
        if small_nbytes:
            dtype = tag & 0xFFFF
            payload = self.data[self.offset + 4 : self.offset + 4 + small_nbytes]
            self.offset += 8
            return dtype, payload

        dtype, nbytes = struct.unpack_from("<II", self.data, self.offset)
        self.offset += 8
        payload = self.data[self.offset : self.offset + nbytes]
        self.offset += nbytes + ((8 - nbytes % 8) % 8)
        return dtype, payload


def _parse_mat_matrix(payload: bytes) -> tuple[str, Any]:
    reader = MatReader(payload)

    _, flags = reader.element()
    matrix_class = int(np.frombuffer(flags[:4], dtype="<u4")[0] & 0xFF)

    _, dims_payload = reader.element()
    dims = tuple(int(v) for v in np.frombuffer(dims_payload, dtype="<i4"))

    _, name_payload = reader.element()
    name = name_payload.decode("latin1", errors="ignore").rstrip("\x00")

    if matrix_class == MX_DOUBLE:
        dtype, numeric_payload = reader.element()
        values = np.frombuffer(numeric_payload, dtype=np.dtype(MAT_DTYPE[dtype])).copy()
        return name, values.reshape(dims, order="F")

    if matrix_class == MX_CHAR:
        dtype, char_payload = reader.element()
        encoding = {
            MI_INT8: "utf-8",
            MI_UINT8: "utf-8",
            MI_UTF8: "utf-8",
            MI_INT16: "utf-16le",
            MI_UINT16: "utf-16le",
            MI_UTF16: "utf-16le",
            MI_UTF32: "utf-32le",
        }.get(dtype, "utf-8")
        return name, char_payload.decode(encoding, errors="ignore").rstrip("\x00")

    if matrix_class == MX_CELL:
        values = []
        while reader.offset < len(reader.data):
            element = reader.element()
            if element is None:
                break
            dtype, cell_payload = element
            if dtype != MI_MATRIX:
                raise ValueError(f"Unexpected MATLAB cell element type: {dtype}")
            values.append(_parse_mat_matrix(cell_payload)[1])
        cell = np.empty(len(values), dtype=object)
        cell[:] = values
        return name, cell.reshape(dims, order="F")

    if matrix_class == MX_STRUCT:
        _, field_len_payload = reader.element()
        field_len = int(np.frombuffer(field_len_payload, dtype="<i4")[0])
        _, field_names_payload = reader.element()
        nfields = len(field_names_payload) // field_len
        field_names = [
            field_names_payload[i * field_len : (i + 1) * field_len]
            .split(b"\x00", 1)[0]
            .decode("latin1")
            for i in range(nfields)
        ]

        nstructs = int(np.prod(dims)) if dims else 1
        structs = [dict() for _ in range(nstructs)]
        for struct_index in range(nstructs):
            for field_name in field_names:
                element = reader.element()
                if element is None:
                    raise ValueError("Unexpected end of MATLAB struct data")
                dtype, field_payload = element
                if dtype != MI_MATRIX:
                    raise ValueError(f"Unexpected MATLAB struct field type: {dtype}")
                structs[struct_index][field_name] = _parse_mat_matrix(field_payload)[1]

        if nstructs == 1:
            return name, structs[0]
        return name, np.array(structs, dtype=object).reshape(dims, order="F")

    raise ValueError(f"Unsupported MATLAB matrix class: {matrix_class}")


def loadmat_v5(path: Path) -> dict[str, Any]:
    """Fallback loader for the MATLAB v5 files used by this example."""

    raw = path.read_bytes()
    if raw[126:128] != b"IM":
        raise ValueError(f"{path} is not a little-endian MATLAB v5 file")

    reader = MatReader(raw[128:])
    variables: dict[str, Any] = {}
    while reader.offset < len(reader.data):
        element = reader.element()
        if element is None:
            break
        dtype, payload = element
        if dtype == MI_COMPRESSED:
            compressed = MatReader(zlib.decompress(payload))
            dtype, payload = compressed.element()
        if dtype != MI_MATRIX:
            continue
        name, value = _parse_mat_matrix(payload)
        variables[name] = value
    return variables


def loadmat(path: Path) -> dict[str, Any]:
    if sio is not None:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return loadmat_v5(path)


def get_field(struct_obj: Any, field_name: str) -> Any:
    if isinstance(struct_obj, dict):
        return struct_obj[field_name]
    return getattr(struct_obj, field_name)


def as_list(value: Any) -> list[Any]:
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [arr.item()]
    return [item for item in arr.ravel(order="F")]


def load_phase_data(base_dir: Path, filename: str) -> tuple[list[np.ndarray], list[str]]:
    mat = loadmat(base_dir / filename)
    adata = mat["adata"]
    phases = [np.asarray(phase, dtype=float) for phase in as_list(get_field(adata, "Phase"))]
    names = [str(name) for name in as_list(get_field(adata, "Name"))]
    return phases, names


def le_fit(data: np.ndarray, terms: int = 10) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Fit low-order Cartesian Legendre-like terms to ``data``."""

    rows, cols = data.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x_grid, y_grid = np.meshgrid(x, y)

    basis = [
        np.ones((rows, cols)),
        x_grid,
        y_grid,
        (3 * x_grid**2 - 1) / 2,
        x_grid * y_grid,
        (3 * y_grid**2 - 1) / 2,
        (5 * x_grid**3 - 3 * x_grid) / 2,
        y_grid * (3 * x_grid**2 - 1) / 2,
        x_grid * (3 * y_grid**2 - 1) / 2,
        (5 * y_grid**3 - 3 * y_grid) / 2,
    ]

    design = np.column_stack([term.ravel() for term in basis])
    values = data.ravel()
    valid = np.isfinite(values)
    if valid.sum() < terms:
        raise ValueError("Not enough finite samples for Legendre fit")

    coeffs, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    fitted = sum(coeffs[i] * basis[i] for i in range(min(terms, len(basis))))
    return fitted, coeffs, basis


def crop_invalid_border(data: np.ndarray) -> np.ndarray:
    valid_rows = ~np.all(~np.isfinite(data), axis=1)
    valid_cols = ~np.all(~np.isfinite(data), axis=0)
    return data[valid_rows][:, valid_cols]


def finite_for_plot(data: np.ndarray) -> np.ndarray:
    output = np.asarray(data, dtype=float).copy()
    output[~np.isfinite(output)] = np.nan
    return output


def fill_nonfinite(data: np.ndarray) -> np.ndarray:
    output = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(output)
    output[~finite] = np.nanmean(output[finite]) if finite.any() else 0.0
    return output


def interp2_regular(data: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a 1-based regular pixel grid."""

    clean = fill_nonfinite(data)
    source_rows = np.arange(1, clean.shape[0] + 1, dtype=float)
    source_cols = np.arange(1, clean.shape[1] + 1, dtype=float)

    temp = np.empty((len(rows), clean.shape[1]), dtype=float)
    for col_index in range(clean.shape[1]):
        temp[:, col_index] = np.interp(rows, source_rows, clean[:, col_index])

    output = np.empty((len(rows), len(cols)), dtype=float)
    for row_index in range(len(rows)):
        output[row_index] = np.interp(cols, source_cols, temp[row_index])
    return output


def make_coordinate_grid(data: np.ndarray, pixsize: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(1, data.shape[0] + 1) * pixsize
    x = np.arange(1, data.shape[1] + 1) * pixsize
    return np.meshgrid(x, y)


def save_surface_grid(
    phases: list[np.ndarray],
    names: list[str],
    pixsize: float,
    output_path: Path,
    title: str,
    ncols: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    if not phases:
        return

    nrows = int(np.ceil(len(phases) / ncols))
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    for index, (phase, name) in enumerate(zip(phases, names), start=1):
        data = finite_for_plot(phase.T)
        xx, yy = make_coordinate_grid(data, pixsize)
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
        ax.plot_surface(xx, yy, data, cmap="viridis", edgecolor="none", linewidth=0)
        ax.set_title(f"direction = {name}")
        ax.set_xlabel("Mirror length direction (mm)")
        ax.set_ylabel("Mirror width direction (mm)")
        ax.set_box_aspect((data.shape[1], data.shape[0], max(data.shape) / 4))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def save_surface_grid_2D(
    phases: list[np.ndarray],
    names: list[str],
    pixsize: float,
    output_path: Path,
    title: str,
    ncols: int = 3,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not phases:
        return

    nrows = int(np.ceil(len(phases) / ncols))
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    
    for index, (phase, name) in enumerate(zip(phases, names), start=1):
        data = finite_for_plot(phase.T)
        xx, yy = make_coordinate_grid(data, pixsize)
        
        # Removed projection="3d"
        ax = fig.add_subplot(nrows, ncols, index)
        
        # Use pcolormesh for 2D grid plotting (or ax.imshow with extent)
        im = ax.pcolormesh(xx, yy, data, cmap="viridis", shading="auto")
        
        # Add a colorbar to show the data scale
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Height")
        
        ax.set_title(f"direction = {name}")
        ax.set_xlabel("Mirror length direction (mm)")
        ax.set_ylabel("Mirror width direction (mm)")
        
        # Set equal aspect ratio so the image isn't stretched artificially
        ax.set_aspect("equal")
        
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def process_sub_apertures(
    phases: list[np.ndarray], names: list[str], pixsize: float
) -> list[dict[str, Any]]:
    sub_apertures = []
    for phase, name in zip(phases, names):
        cropped = crop_invalid_border(phase)
        zz = finite_for_plot(cropped.T)
        xx, yy = make_coordinate_grid(zz, pixsize)
        sub_apertures.append(
            {
                "XX": xx,
                "YY": yy,
                "ZZ": zz,
                "center_x": (zz.shape[1] + 1) / 2,
                "center_y": (zz.shape[0] + 1) / 2,
                "Name": name,
            }
        )
    return sub_apertures


def save_sub_aperture_comparison(sub_apertures: list[dict[str, Any]], output_path: Path) -> None:
    if len(sub_apertures) < 3:
        return

    import matplotlib.pyplot as plt

    first = sub_apertures[0]
    third = sub_apertures[2]
    rotated = np.fliplr(first["ZZ"].T)
    rows = min(third["ZZ"].shape[0], rotated.shape[0])
    cols = min(third["ZZ"].shape[1], rotated.shape[1])
    difference = third["ZZ"][:rows, :cols] - rotated[:rows, :cols]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        (first["ZZ"], first["Name"]),
        (third["ZZ"], third["Name"]),
        (difference, f"{third['Name']} minus {first['Name']} rotated 90 deg"),
    ]
    for ax, (data, title) in zip(axes, panels):
        image = ax.imshow(data, cmap="viridis", origin="lower", vmin=-20, vmax=20)
        ax.set_title(title)
        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_zone_demo(sub_aperture: dict[str, Any], output_path: Path, zone_size: int) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    test_fov = sub_aperture["ZZ"]
    rows = np.linspace(1, test_fov.shape[0], zone_size * 2)
    cols = np.linspace(1, test_fov.shape[1], zone_size * 2)
    zz_reduce = interp2_regular(test_fov, rows, cols)

    zones = {
        "1st zone": zz_reduce[zone_size:, zone_size:],
        "2nd zone": zz_reduce[zone_size:, :zone_size],
        "3rd zone": zz_reduce[:zone_size, :zone_size],
        "4th zone": zz_reduce[:zone_size, zone_size:],
    }
    vrange = [-100, 100]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (name, zone) in zip(axes.ravel(), zones.items()):
        # image = ax.imshow(zone, cmap="viridis", origin="lower", vmin=vrange[0], vmax=vrange[1])
        image = ax.imshow(zone, cmap="viridis", origin="lower")
        ax.set_title(name)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    # image = axes.ravel()[4].imshow(zz_reduce, cmap="viridis", origin="lower", vmin=vrange[0], vmax=vrange[1])
    image = axes.ravel()[4].imshow(zz_reduce, cmap="viridis", origin="lower")
    axes.ravel()[4].set_title("entire zone")
    fig.colorbar(image, ax=axes.ravel()[4], fraction=0.046, pad=0.04)
    axes.ravel()[5].axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {"resampled": zz_reduce, "zones": zones}


def run(args: argparse.Namespace) -> None:
    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases_36, names_36 = load_phase_data(base_dir, "circ36.mat")
    phases_rot1, names_rot1 = load_phase_data(base_dir, "circ_rot_1.mat")
    phases_rot2, names_rot2 = load_phase_data(base_dir, "circ_rot_2.mat")

    print('data 1: {}:\n {}\nName: {}\n'.format("circ36", np.array(phases_36).shape, names_36))
    print('data 2: {}:\n {}\nName: {}\n'.format("circ_rot_1", np.array(phases_rot1).shape, names_rot1))
    print('data 3: {}:\n {}\nName: {}\n'.format("circ_rot_2", np.array(phases_rot2).shape, names_rot2))

    save_surface_grid_2D(phases_36,
        names_36,
        args.pixsize,
        output_dir / "circ36_surfaces.png",
        "circ36 reference measurements",
        ncols=max(1, len(phases_36)),
        )
    # save_surface_grid(
    #     phases_36,
    #     names_36,
    #     args.pixsize,
    #     output_dir / "circ36_surfaces.png",
    #     "circ36 reference measurements",
    #     ncols=max(1, len(phases_36)),
    # )

    sub_apertures_1 = process_sub_apertures(phases_rot1, names_rot1, args.pixsize)
    sub_apertures_2 = process_sub_apertures(phases_rot2, names_rot2, args.pixsize)
    # save_surface_grid(
    #     [item["ZZ"].T for item in sub_apertures],
    #     [item["Name"] for item in sub_apertures],
    #     args.pixsize,
    #     output_dir / "circ_rot_2_sub_apertures.png",
    #     "circ_rot_2 cropped sub-apertures",
    # )
    
    save_surface_grid_2D(
        [item["ZZ"].T for item in sub_apertures_1],
        [item["Name"] for item in sub_apertures_1],
        args.pixsize,
        output_dir / "circ_rot_1_sub_apertures.png",
        "circ_rot_1 cropped sub-apertures",
    )
    save_sub_aperture_comparison(sub_apertures_1, output_dir / "rotated_difference_rot1.png")
    
    save_surface_grid_2D(
        [item["ZZ"].T for item in sub_apertures_2],
        [item["Name"] for item in sub_apertures_2],
        args.pixsize,
        output_dir / "circ_rot_2_sub_apertures.png",
        "circ_rot_1 cropped sub-apertures",
    )
    save_sub_aperture_comparison(sub_apertures_2, output_dir / "rotated_difference_rot2.png")

    zone_size = args.zone_size
    if zone_size is None and sub_apertures_1:
        zone_size = min(198 * 4, sub_apertures_1[0]["ZZ"].shape[0] // 2, sub_apertures_1[0]["ZZ"].shape[1] // 2)
    if sub_apertures_1 and zone_size and zone_size > 0:
        zone_result = save_zone_demo(sub_apertures_1[0], output_dir / "zone_resampling_1.png", zone_size)
        np.savez_compressed(
            output_dir / "zone_resampling_1.npz",
            resampled=zone_result["resampled"],
            **{key.replace(" ", "_"): value for key, value in zone_result["zones"].items()},
        )

    zone_size = args.zone_size
    if zone_size is None and sub_apertures_2:
        zone_size = min(198 * 4, sub_apertures_2[0]["ZZ"].shape[0] // 2, sub_apertures_2[0]["ZZ"].shape[1] // 2)
    if sub_apertures_2 and zone_size and zone_size > 0:
        zone_result = save_zone_demo(sub_apertures_2[0], output_dir / "zone_resampling_2.png", zone_size)
        np.savez_compressed(
            output_dir / "zone_resampling_2.npz",
            resampled=zone_result["resampled"],
            **{key.replace(" ", "_"): value for key, value in zone_result["zones"].items()},
        )

    if phases_36:
        reference = crop_invalid_border(phases_36[0])
        reference[~np.isfinite(reference)] = np.nan
        valid_cols = ~np.any(~np.isfinite(reference), axis=0)
        reference = reference[:, valid_cols].T
        fitted, coeffs, basis = le_fit(reference, terms=10)
        residual = reference - coeffs[0] * basis[0] - coeffs[3] * basis[3]

        curvature = np.inf
        if abs(coeffs[3]) > np.finfo(float).eps:
            curvature = (args.pixsize * reference.shape[1] / 2) ** 2 / 3 / (coeffs[3] * 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, data, title in [
            (axes[0], reference, "reference"),
            (axes[1], fitted, "Legendre fit"),
            (axes[2], residual, "without piston/curvature"),
        ]:
            image = ax.imshow(data, cmap="viridis", origin="lower")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(output_dir / "legendre_fit.png", dpi=160)
        plt.close(fig)

        print(f"Loaded {len(phases_36)} circ36 phase map(s) and {len(phases_rot2)} rotated phase map(s).")
        print(f"Legendre coefficient[3] curvature term: {coeffs[3]:.6g}")
        print(f"Coefficient of twist term: {coeffs[4]:.6g}")
        print(f"R = {curvature:.6g} mm")

    print(f"Results written to: {output_dir.resolve()}")
    if args.show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="circle_ref", help="Directory containing circ*.mat files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for figures and arrays.")
    parser.add_argument("--pixsize", type=float, default=0.177, help="Pixel size in mm.")
    parser.add_argument("--zone-size", type=int, default=None, help="Quadrant size after resampling.")
    parser.add_argument("--show", action="store_true", help="Show Matplotlib windows in addition to saving files.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
