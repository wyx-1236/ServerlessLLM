"""Compatibility module for the Store-owned preparation runtime."""

import sys

from sllm_store import vllm_preparation as _implementation

sys.modules[__name__] = _implementation
