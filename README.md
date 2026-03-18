# cgNAplusParams

Python interface to the [cgDNA+](#ref-cgnaplus) and cgRNA+ coarse-grained nucleic acid model parameter sets. Given a DNA or RNA sequence it returns the sequence-dependent groundstate rigid-body parameters and stiffness matrix, and can build a full SE(3) pose chain for 3-D structure visualisation.

## What it does

- **`constructSeqParms(sequence, parameter_set_name)`** — assembles the cgDNA+/cgRNA+ groundstate vector and stiffness matrix for an arbitrary sequence.
- **`cgnaplus2rbp(sequence, ...)`** — convenience wrapper that returns the base-pair-step rigid-body parameters (`gs`) and optionally the stiffness matrix in physical units.
- **`rbp_conf(rbp_params, ...)`** — Build rigid base pair configuration: converts a sequence of SE(3) rigid-body parameters into a chain of 4×4 homogeneous transformation matrices (one per base pair).
- **`gen_pdb(...)` / `visualize_chimerax(...)`** — export a coarse-grained structure as PDB or a ChimeraX session.

## Installation

Two dependency libraries are required: **SO3** and **pyConDec**.
They are bundled as git submodules so you can choose how to provide them.

### Option A — recursive clone (submodule workflow)

```bash
git clone --recurse-submodules https://github.com/eskoruppa/cgNAplusParams.git
cd cgNAplusParams
pip install .
```

If you already have a plain clone, initialise the submodules afterwards:

```bash
git submodule update --init --recursive
pip install .
```

No extra steps are needed — `cgnaplusparams/_so3.py` and
`cgnaplusparams/_pycondec.py` detect the local checkouts automatically and
add them to the Python path.

### Option B — pip (no submodules required)

```bash
pip install "cgnaplusparams[pip] @ git+https://github.com/eskoruppa/cgNAplusParams.git"
```

The `[pip]` extra fetches SO3 and pyConDec from GitHub automatically.
You can also install each piece individually:

```bash
pip install git+https://github.com/eskoruppa/SO3.git
pip install git+https://github.com/eskoruppa/pyConDec.git
pip install git+https://github.com/eskoruppa/cgNAplusParams.git
```

## Quick start

```python
import numpy as np
from cgnaplusparams import cgnaplus2rbp, rbp_conf

sequence = "ATACGCTTGCATGC"

# Groundstate rigid-body parameters and stiffness matrix
result = cgnaplus2rbp(sequence,include_stiffness=True)
gs     = result["gs"]      # (nbp-1, 6) rigid-body parameters
stiff  = result["stiff"]   # (6*(nbp-1), 6*(nbp-1)) stiffness matrix

# Chain of 3D poses →  (nbp, 4, 4) SE(3) matrices
conf = rbp_conf(gs)
```

### Available parameter sets

| `parameter_set_name` | System |
|---|---|
| `Prmset_cgDNA+_CGF_10mus_int_12mus_ends` *(default)* | DNA, TIP3P/JC/BSC1 |
| `cgDNA+_Curves_BSTJ_10mus_FS` | DNA, Curves+ frames |
| `Prmset_cgRNA+_OL3_CGF_10mus_int_12mus_ends` | RNA |
| `cgHYB+_CGF_OL3_BSC1_10mus_FS_GC_ends` | DNA–RNA hybrid |
| `Di_hmethyl_methylated-hemi_combine` | Methylated DNA |

## References

1. <a name="ref-cgnaplus"></a>**R. Sharma, J. H. Maddocks, and others**, *cgDNA+: A sequence-dependent coarse-grain model of double-stranded DNA*, (2023).

## License

GNU General Public License v2 — see [LICENSE](LICENSE).