"""Auto-import bootstrap for moe_ascend_npu SGLang monkey patches.

A ``.pth`` file (``moe_ascend_npu.pth``) auto-imports this module at interpreter
start-up. Importing this module installs a lightweight ``builtins.__import__``
wrapper that waits until ``sglang.srt`` has been imported AND the outermost
import has completed (import depth back to 0), then applies the patches
(guarded by ``is_npu()``) and unwraps itself.

Firing at depth 0 (rather than the moment ``sglang.srt`` appears in
``sys.modules``) is essential: ``sglang.srt`` is added to ``sys.modules`` before
its ``__init__`` finishes, so patching mid-import triggers circular-import
errors. At depth 0 the whole SGLang import tree is settled.

This avoids importing SGLang (or even torch) in non-SGLang processes: the
wrapper only does a cheap ``_TRIGGER in sys.modules`` check at depth 0 until it
fires.
"""

from __future__ import annotations

import builtins
import logging
import sys
import warnings

logger = logging.getLogger(__name__)

_TRIGGER = "sglang.srt"
_done = False
_real_import = builtins.__import__


def _install_hook() -> None:
    depth = 0

    def _wrapped_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal depth
        depth += 1
        try:
            mod = _real_import(name, globals, locals, fromlist, level)
        finally:
            depth -= 1

        global _done
        if not _done and depth == 0 and _TRIGGER in sys.modules:
            _done = True
            # Unwrap first so re-entrant imports during patching are unaffected.
            builtins.__import__ = _real_import
            try:
                from moe_ascend_npu.patches import apply_patches

                apply_patches()
            except Exception as exc:  # pragma: no cover - never break startup
                warnings.warn(
                    f"moe_ascend_npu: failed to apply SGLang patches: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return mod

    builtins.__import__ = _wrapped_import


_install_hook()
