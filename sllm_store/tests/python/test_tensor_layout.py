"""Real CPU Torch state/alias tests, separate from light CPU contracts."""

import copy
import importlib
import importlib.metadata
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

torch = pytest.importorskip(
    "torch", reason="requires separate CPU Torch environment"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_cpu_tensor_profile_matches_declared_roots():
    inputs = (
        Path(__file__).resolve().parents[3]
        / "tests/baseline/requirements-tensor-cpu.in"
    )
    for line in inputs.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        assert (
            importlib.metadata.version(requirement.name)
            in requirement.specifier
        )
    assert torch.version.cuda is None
    assert not torch.cuda.is_initialized()


class Toy(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset
        self.weight = torch.nn.Parameter(base)
        self.tied = self.weight
        self.transposed = torch.nn.Parameter(base.t())
        self.slice = torch.nn.Parameter(base.flatten()[2:5])
        self.register_buffer("running", base.flatten()[1:4])
        self.register_buffer(
            "counter", torch.tensor(7 + offset, dtype=torch.int64)
        )
        self.register_buffer(
            "scratch", torch.tensor([5], dtype=torch.int32), persistent=False
        )


def api():
    return importlib.import_module("sllm_store.tensor_layout")


def test_complete_state_roundtrip_preserves_views_buffers_and_aliases():
    layout = api()
    source = Toy(100)
    storages, metadata = layout.capture_state(source)
    assert len(storages) == 2
    assert metadata["tensors"]["counter"]["kind"] == "buffer"
    assert "scratch" not in metadata["tensors"]
    target = Toy()
    layout.bind_state(target, storages, metadata, device="cpu")
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name
    assert target.weight is target.tied
    assert target.transposed.stride() == (1, 4)
    assert target.slice.storage_offset() == 2
    assert (
        target.running.untyped_storage().data_ptr()
        == target.weight.untyped_storage().data_ptr()
    )
    assert target.scratch.dtype == torch.int32
    with torch.no_grad():
        target.slice[0] = -17
    assert target.weight.flatten()[2] == -17
    assert target._sllm_storage_owners is storages


@pytest.mark.parametrize(
    "change", ["missing", "extra", "shape", "dtype", "offset", "alias"]
)
def test_invalid_layout_fails_before_changing_target(change):
    layout = api()
    storages, metadata = layout.capture_state(Toy(100))
    metadata = copy.deepcopy(metadata)
    tensors = metadata["tensors"]
    if change == "missing":
        tensors.pop("counter")
    elif change == "extra":
        tensors["extra"] = copy.deepcopy(tensors["counter"])
    elif change == "shape":
        tensors["weight"]["shape"] = [4, 3]
    elif change == "dtype":
        tensors["counter"]["dtype"] = "float32"
    elif change == "offset":
        tensors["slice"]["storage_offset"] = 1000
    else:
        tensors["tied"]["storage_offset"] = 1
    target = Toy()
    before = {
        name: tensor.clone() for name, tensor in target.state_dict().items()
    }
    with pytest.raises(ValueError):
        layout.bind_state(target, storages, metadata, device="cpu")
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_storage_corruption_fails_validation():
    layout = api()
    storages, metadata = layout.capture_state(Toy(100))
    key = next(iter(storages))
    storages[key] = storages[key].clone()
    storages[key][0] ^= 1
    with pytest.raises(ValueError, match="digest"):
        layout.bind_state(Toy(), storages, metadata, device="cpu")


def test_persistent_binding_wins_over_nonpersistent_object_alias():
    layout = api()
    source, target = Toy(100), Toy()
    source.register_buffer("ephemeral_alias", source.running, persistent=False)
    target.register_buffer("ephemeral_alias", target.running, persistent=False)
    # A device transfer materializes a separate tensor; simulate that boundary
    # with an actual CPU clone, without starting CUDA for this unit test.
    target.running.to = lambda **kwargs: target.running.clone()
    storages, metadata = layout.capture_state(source)
    layout.bind_state(target, storages, metadata, device="cpu")
    assert torch.equal(target.running, source.running)
    assert target.ephemeral_alias is target.running


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float64, torch.bool]
)
def test_scalar_empty_and_mixed_dtype_state(dtype):
    layout = api()
    source, target = torch.nn.Module(), torch.nn.Module()
    for model, value in [(source, 1), (target, 0)]:
        model.register_buffer("scalar", torch.tensor(value, dtype=dtype))
        model.register_buffer("empty", torch.empty((0, 3), dtype=dtype))
        model.register_buffer(
            "integer", torch.tensor([value], dtype=torch.int64)
        )
    storages, metadata = layout.capture_state(source)
    layout.bind_state(target, storages, metadata, device="cpu")
    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name])
        assert value.dtype == target.state_dict()[name].dtype
