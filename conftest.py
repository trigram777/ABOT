"""Make the sample dataset available before anything imports `lab.session`.

The dataset is NOT committed. It is ~70 MB of parquet, it is fully determined
by a seed, and a generator is smaller than its own output -- so the repository
carries `tools/make_sample_data.py` and builds the data on first use instead.

Set LAB_DATA_ROOT to run against a real production store instead; then nothing
is generated and the `realdata` tests stop skipping.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample_data"
STAMP = SAMPLE / "dataset_catalog" / "master_session_index.parquet"


def pytest_configure(config):
    if os.environ.get("LAB_DATA_ROOT") or STAMP.exists():
        return
    print("\nbuilding the synthetic sample dataset (once, a few seconds)…", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_sample_data.py")],
                   check=True)


def pytest_collection_modifyitems(config, items):
    """Skip the tests that assert facts about the PRODUCTION dataset.

    They are not broken and the sample data is not deficient — they check
    things only the real panel can carry: its size, a session named by date,
    or a base rate of the real market. Marking them is how the suite stays
    honest about the difference. Point LAB_DATA_ROOT at the real store and
    they run.
    """
    import os
    import pytest
    if os.environ.get("LAB_DATA_ROOT"):
        return
    skip = pytest.mark.skip(reason="asserts a fact about the production dataset; "
                                   "set LAB_DATA_ROOT to run it")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip)
