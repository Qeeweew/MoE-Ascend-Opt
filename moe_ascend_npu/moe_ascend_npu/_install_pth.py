"""Install/uninstall the ``moe_ascend_npu.pth`` auto-import hook.

Usage::

    python -m moe_ascend_npu._install_pth          # install
    python -m moe_ascend_npu._install_pth remove   # uninstall

The ``.pth`` file (shipped next to this module) contains a single ``import
moe_ascend_npu._bootstrap`` line. When placed in a directory on ``sys.path``
that ``site`` processes (i.e. site-packages), Python executes it at every
interpreter start-up, installing the lazy SGLang monkey-patch hook.

``build_kernels.sh`` runs this automatically after building the kernels.
"""

from __future__ import annotations

import shutil
import site
import sys
import sysconfig
from pathlib import Path

PTH_NAME = "moe_ascend_npu.pth"


def _pth_source() -> Path:
    # moe_ascend_npu/_install_pth.py -> ../moe_ascend_npu.pth (project root)
    return Path(__file__).resolve().parents[1] / PTH_NAME


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    if hasattr(site, "getsitepackages"):
        dirs.extend(Path(d) for d in site.getsitepackages())
    try:
        dirs.append(Path(sysconfig.get_paths().get("purelib", "")))
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        dirs.append(Path(site.getusersitepackages()))
    # Deduplicate (getsitepackages and purelib often overlap) while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        try:
            d = d.resolve()
        except Exception:
            pass
        if d and d.exists() and d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def install() -> int:
    src = _pth_source()
    if not src.exists():
        print(f"[install_pth] source .pth not found at {src}", file=sys.stderr)
        return 1

    targets = _candidate_dirs()
    if not targets:
        print("[install_pth] no site-packages directory found", file=sys.stderr)
        return 1

    dst = targets[0] / PTH_NAME
    shutil.copyfile(src, dst)
    print(f"[install_pth] installed {PTH_NAME} -> {dst}")
    # Remove stale copies in other site dirs (skip the one we just wrote) to
    # avoid confusion.
    for d in targets[1:]:
        stale = d / PTH_NAME
        if stale.exists():
            stale.unlink()
    return 0


def remove() -> int:
    removed = False
    for d in _candidate_dirs():
        p = d / PTH_NAME
        if p.exists():
            p.unlink()
            print(f"[install_pth] removed {p}")
            removed = True
    if not removed:
        print("[install_pth] nothing to remove")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("remove", "uninstall"):
        return remove()
    return install()


if __name__ == "__main__":
    raise SystemExit(main())
