import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def project_root(monkeypatch):
    # The app resolves configs/ and data/ relative to the working directory.
    monkeypatch.chdir(ROOT)
    return ROOT
