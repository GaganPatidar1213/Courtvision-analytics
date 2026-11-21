import pytest

from utils.data_loader import load_archive
from utils.feature_engineering import build_windows
from utils.preprocessing import clean


@pytest.fixture(scope="session")
def basketball():
    raw = load_archive()
    processed, audit, missing = clean(raw)
    return raw, processed, audit, missing, build_windows(processed)
