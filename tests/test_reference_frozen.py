"""The vendored reference must stay byte-identical to upstream commit faa93c7."""

import hashlib
from pathlib import Path

import pytest

import ace_reference

EXPECTED_SHA256 = {
    "hardl1ace.py": "fa8513bd18c0dcc8e17e64318b4ccd85b72405a745bcfe7332466a9d7fab6da6",
    "softl1ace.py": "815d2acd230fe0503f8be0ad45b16c445200df9f7c98aa1549c538392d86eb8e",
}


@pytest.mark.parametrize("filename", sorted(EXPECTED_SHA256))
def test_reference_file_unchanged(filename: str) -> None:
    path = Path(ace_reference.__file__).parent / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[filename]
