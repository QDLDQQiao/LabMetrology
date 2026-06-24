import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_asc(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # Header ends at the first blank line
    blank = next(i for i, line in enumerate(lines) if line.strip() == "")

    meta = {}
    for line in lines[:blank]:
        parts = line.split("\t")
        if len(parts) >= 2:
            key = parts[0].strip()
            value = parts[1].strip()
            unit = parts[2].strip() if len(parts) >= 3 else ""
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except Exception:
                pass
            meta[key] = value
            if unit:
                meta[key + " Unit"] = unit

    z = np.genfromtxt(
        path,
        delimiter="\t",
        skip_header=blank + 1,
        dtype=float,
        invalid_raise=False,
        autostrip=True,
    )

    # Remove trailing empty column caused by final tab
    if z.ndim == 2 and np.all(~np.isfinite(z[:, -1])):
        z = z[:, :-1]

    # "Bad" becomes NaN; also remove inf
    z[~np.isfinite(z)] = np.nan

    return z, meta


def load_txt(path):
    z = np.genfromtxt(
        path,
        delimiter="\t",
        skip_header=1,
        dtype=float,
        invalid_raise=False,
    )

    if z.ndim == 2 and np.all(~np.isfinite(z[:, -1])):
        z = z[:, :-1]

    z[~np.isfinite(z)] = np.nan
    return z


def plot_map(z, title="Height map", unit="nm"):
    plt.figure(figsize=(8, 5.5))
    im = plt.imshow(z, origin="upper", aspect="equal")
    plt.title(title)
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    cbar = plt.colorbar(im)
    cbar.set_label(unit)
    plt.tight_layout()
    plt.show()

def load_hll_this_format(path, nx=1280, ny=960, offset=624):
    """
    Experimental reader for this uploaded .hll file.

    The file contains three little-endian float32 maps after byte 624.
    Very large values, around 3.4e38, are invalid pixels.
    """
    raw = np.fromfile(path, dtype="<f4", offset=offset)

    pixels = nx * ny
    n_maps = raw.size // pixels

    raw = raw[: n_maps * pixels]
    maps = raw.reshape(n_maps, ny, nx).astype(float)

    maps[maps > 1e30] = np.nan

    return [maps[i] for i in range(n_maps)]

z_asc, meta = load_asc("CM2_4mm-3.asc")

print("Metadata:")
for k, v in meta.items():
    print(k, "=", v)

print("Shape:", z_asc.shape)
print("Finite pixels:", np.isfinite(z_asc).sum())
print("Min:", np.nanmin(z_asc))
print("Mean:", np.nanmean(z_asc))
print("Max:", np.nanmax(z_asc))
print("PV:", np.nanmax(z_asc) - np.nanmin(z_asc))
print("RMS:", np.sqrt(np.nanmean((z_asc - np.nanmean(z_asc)) ** 2)))

plot_map(z_asc, "ASC height map", unit=meta.get("Heigh Unit", "nm"))

maps = load_hll_this_format("CM2_4mm-3.hll")

for i, m in enumerate(maps):
    print("Map", i)
    print("  shape:", m.shape)
    print("  finite:", np.isfinite(m).sum())
    print("  min:", np.nanmin(m))
    print("  mean:", np.nanmean(m))
    print("  max:", np.nanmax(m))