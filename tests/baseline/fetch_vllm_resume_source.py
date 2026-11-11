"""Fetch hash-pinned Python source for CPU tests without installing vLLM."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

MANIFEST = (
    Path(__file__).with_name("fixtures") / "vllm-0.31.0-resume-source.json"
)


def verify_source(root, manifest):
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned vLLM source hash mismatch: {name}")


def fetch_source(root):
    root = root.resolve()
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path in manifest: {name}")
        destination = (root / name).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Source path escapes output directory: {name}")
        if destination.is_file():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
                continue
        with urlopen(
            f"{manifest['source_url']}/{name}", timeout=30
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded vLLM source hash mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    verify_source(root, manifest)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(fetch_source(args.output_directory))
