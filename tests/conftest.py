import os
import sys
import random
from pathlib import Path


import numpy as np
import pytest
import torch




@pytest.fixture(autouse=True)
def set_seed_and_determinism():
    """Make runs repeatable and keep cudnn from changing algorithms across runs."""
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer deterministic ops when available (some kernels may still be nondeterministic)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
    except Exception:
        pass
    yield




@pytest.fixture(scope="session", autouse=True)
def add_src_to_syspath():
    """Allow importing packages from a src/ layout without installing.
    If you've done `pip install -e .`, this is harmless; otherwise it ensures
    `import dpcs.dpcs` works.
    """
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    yield