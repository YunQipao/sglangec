"""Lazy bridge to checkout-only KV EC research code. Not imported when disabled."""

import importlib
import importlib.util
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
_capture_module = None


def create_online_protector(scheduler):
    # Reuse the checkout-only package loader without importing experiments on
    # the default inference path. Startup errors intentionally propagate.
    root = Path(__file__).resolve().parents[4] / "experiments" / "kv_ec"
    name = "_sglang_kv_ec_experiment"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    online = importlib.import_module(name + ".online_chunk")
    return online.OnlineProtector.from_env(scheduler)


def create_pause_gate(scheduler):
    """Load the checkout-only gate; failures must propagate, never resume."""
    root = Path(__file__).resolve().parents[4] / "experiments" / "kv_ec"
    name = "_sglang_kv_ec_experiment"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    pause = importlib.import_module(name + ".scheduler_pause")
    return pause.SchedulerPauseGate.from_env(scheduler)


def maybe_capture_prefill(scheduler, batch, req):
    global _capture_module
    try:
        if _capture_module is None:
            # A fixed, trusted path relative to this checkout, not user-provided code.
            root = Path(__file__).resolve().parents[4] / "experiments" / "kv_ec"
            name = "_sglang_kv_ec_experiment"
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    name, root / "__init__.py", submodule_search_locations=[str(root)]
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            _capture_module = importlib.import_module(name + ".debug_capture")
        _capture_module.maybe_capture(scheduler, batch, req)
    except Exception as exc:
        # Never log request text or exception strings containing user data.
        # No distributed calls: local export failure does not skip model collectives.
        logger.warning("KV_EC_DEBUG bridge failed (%s); inference continues", type(exc).__name__)
