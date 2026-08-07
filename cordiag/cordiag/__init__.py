"""
cordiag — BridgeOmics zPG + TG shared diagnostics package.

Module status:
  m1.py          Shared M1 model core (stratum-conditioned Ridge) — migrated
  tg.py          TG diagnostic (tgdecomp/core.py) — migrated
  zpg.py         zPG diagnostic (metrics_v3.py) — migrated
  calibration.py TG simulation calibration suite (tgdecomp/simulation.py) — migrated
  cli.py         CLI entry point — migrated
  plotting.py    Visualization (tgdecomp/visualization.py) — scheduled for future release
                 (package has no plotting dependency; clear error via __getattr__ if needed)

Specifications: tg_migrate_spec, zpg_extract_spec
"""

from . import m1

__version__ = "0.1.0"

__all__ = ["m1", "zpg", "tg", "calibration", "cli"]

# PEP 562 lazy __getattr__: attribute access auto-works after module lands on disk;
# gives clear error for incomplete modules, ensuring `import cordiag` never fails.
_LAZY_MODULES = ("zpg", "tg", "calibration", "cli")


def __getattr__(name):
    if name in _LAZY_MODULES:
        try:
            module = __import__(f"{__name__}.{name}", fromlist=[name])
        except ImportError as exc:
            raise ModuleNotFoundError(
                f"cordiag.{name} is not available yet — it is being migrated in "
                f"parallel; retry once cordiag/{name}.py is in place "
                f"(underlying error: {exc})"
            ) from exc
        globals()[name] = module  # cache
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
