from __future__ import annotations

import fasthep_distributed


def test_package_imports_and_exposes_version() -> None:
    assert isinstance(fasthep_distributed.__version__, str)
    assert fasthep_distributed.__version__
