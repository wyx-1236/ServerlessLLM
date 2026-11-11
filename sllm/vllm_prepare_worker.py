"""Compatibility module for the Store-owned preparation runtime."""

import sys

from sllm_store import vllm_prepare_worker as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
