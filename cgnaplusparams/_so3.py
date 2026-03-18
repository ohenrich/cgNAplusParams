"""
Shim that resolves so3 regardless of whether it is present as a
git-submodule (cgnaplusparams/SO3/) or as a pip-installed package.

Priority:
  1. Local submodule:  cgnaplusparams/SO3/so3  (recursive-clone / dev workflow)
  2. Installed package: so3                     (pip-installed dependency)

Raises ImportError with a helpful message if neither is available.
"""

import sys
import pathlib

# Directory containing the SO3 submodule (sibling of this file)
_SO3_SUBMODULE = pathlib.Path(__file__).parent / "SO3"


def _ensure_so3() -> None:
    """Insert the submodule root into sys.path so that ``import so3`` resolves
    to the local checkout.  Does nothing if so3 is already importable or if
    the submodule is not present."""
    submodule_root = str(_SO3_SUBMODULE)
    if submodule_root not in sys.path:
        sys.path.insert(0, submodule_root)


try:
    # 1. Local submodule: cgnaplusparams/SO3/so3  (dev / recursive-clone workflow)
    if (_SO3_SUBMODULE / "so3").is_dir():
        _ensure_so3()
    import so3  # noqa: E402
except (ImportError, ModuleNotFoundError):
    try:
        # 2. Installed package
        import so3  # noqa: F811
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "so3 could not be found. Either:\n"
            "  • clone cgNAplusParams with --recurse-submodules so that cgnaplusparams/SO3/ is populated:\n"
            "      git clone --recurse-submodules https://github.com/eskoruppa/cgNAplusParams.git\n"
            "    or, for an existing clone:\n"
            "      git submodule update --init --recursive\n"
            "  • or install SO3 via pip:\n"
            "      pip install SO3\n"
            "    or from source:\n"
            "      git clone https://github.com/eskoruppa/SO3.git && cd SO3 && pip install .\n"
        ) from e

__all__ = ["so3"]
