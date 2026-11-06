"""
Integration tests for CLI subcommands.
"""

import subprocess
import sys
import pytest


def run_cli_cmd(*args):
    cmd = [sys.executable, "cli.py"] + list(args)
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res


def test_cli_inspect():
    res = run_cli_cmd("inspect")
    assert res.returncode == 0
    assert "TREE-STRUCTURED SPECULATIVE DECODING ENGINE" in res.stdout


def test_cli_tree():
    res = run_cli_cmd("tree", "--branching", "2,2")
    assert res.returncode == 0
    assert "Node 0" in res.stdout
    assert "Tree Attention Mask Shape" in res.stdout


def test_cli_verify():
    res = run_cli_cmd("verify")
    assert res.returncode == 0
    assert "Accepted Token IDs" in res.stdout
    assert "Emitted Bonus Token" in res.stdout


def test_cli_benchmark():
    res = run_cli_cmd("benchmark", "--tokens", "10")
    assert res.returncode == 0
    assert "Benchmark Summary" in res.stdout
    assert "Autoregressive Target Passes" in res.stdout
