"""
tests/test_package.py
"""

from clinicalxai import __version__
import subprocess


def test_version_is_importable():
    assert isinstance(__version__, str)
    assert __version__


def test_cli_entry_point_registered():
    result = subprocess.run(["clinicalxai", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
