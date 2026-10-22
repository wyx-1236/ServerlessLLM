"""Versioned logical tensor views over Store's existing raw byte format."""

import ctypes
import hashlib
import math


def _byte_digest(tensor):
    import torch

    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    if not raw.numel():
        return hashlib.sha256(b"").hexdigest()
    data = (ctypes.c_ubyte * raw.numel()).from_address(raw.data_ptr())
    return hashlib.sha256(memoryview(data)).hexdigest()


def _bindings(model):
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    result = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor state: {name}")
        if tensor.is_meta or tensor.layout != torch.strided:
            raise ValueError(f"Unsupported tensor layout: {name}")
        if name in parameters:
            result[name] = (parameters[name], "parameter")
        elif name in buffers:
            result[name] = (buffers[name], "buffer")
        else:
            raise ValueError(f"State has no parameter/buffer binding: {name}")
    return result


def capture_state(model):
    """Copy each backing storage to CPU once, retaining all logical aliases.

    Store writes the returned uint8 vectors with its existing save_dict API.
    The sidecar records the names, dtypes, views and parameter/buffer roles.
    """
    import torch

    storages = {}
    storage_ids = {}
    layout = {"schema_version": 1, "storages": {}, "tensors": {}}
    for name, (tensor, kind) in _bindings(model).items():
        storage = tensor.untyped_storage()
        # Storage identity also distinguishes independent empty tensors.
        identity = (str(tensor.device), storage._cdata)
        if identity not in storage_ids:
            key = f"storage_{len(storage_ids)}"
            storage_ids[identity] = key
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            raw = raw.set_(storage, 0, (storage.nbytes(),), (1,)).cpu()
            storages[key] = raw
            layout["storages"][key] = {
                "nbytes": raw.numel(),
                "sha256": _byte_digest(raw),
            }
        key = storage_ids[identity]
        view = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            storages[key].untyped_storage(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        layout["tensors"][name] = {
            "kind": kind,
            "storage": key,
            "storage_offset": tensor.storage_offset(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "sha256": _byte_digest(view),
        }
    return storages, layout


def bind_state(model, storages, layout, *, device):
    """Validate the entire layout before binding any parameter or buffer.

    Raw Store tensors own the CUDA allocation, so the model must retain them
    for the complete lifetime of every restored view.
    """
    import torch

    if layout.get("schema_version") != 1:
        raise ValueError("Unsupported tensor layout schema")
    records = layout.get("tensors", {})
    storage_records = layout.get("storages", {})
    bindings = _bindings(model)
    if set(records) != set(bindings):
        raise ValueError("Tensor layout key mismatch")
    if set(storage_records) != set(storages):
        raise ValueError("Storage layout key mismatch")
    if {record["storage"] for record in records.values()} != set(storages):
        raise ValueError("Unreferenced or missing tensor storage")
    target_device = torch.device(device)
    for key, raw in storages.items():
        record = storage_records[key]
        if (
            raw.dtype != torch.uint8
            or raw.ndim != 1
            or not raw.is_contiguous()
            or raw.device != target_device
            or raw.numel() != record["nbytes"]
        ):
            raise ValueError(f"Invalid raw storage: {key}")
        if _byte_digest(raw) != record["sha256"]:
            raise ValueError(f"Storage digest mismatch: {key}")
    views = {}
    objects = {}
    for name, (target, kind) in bindings.items():
        record = records[name]
        shape, stride = record["shape"], record["stride"]
        offset = record["storage_offset"]
        if (
            record["kind"] != kind
            or shape != list(target.shape)
            or stride != list(target.stride())
            or record["dtype"] != str(target.dtype)
        ):
            raise ValueError(f"Tensor layout mismatch: {name}")
        if (
            type(offset) is not int
            or offset < 0
            or any(type(x) is not int or x < 0 for x in shape + stride)
        ):
            raise ValueError(f"Invalid tensor bounds: {name}")
        required = offset
        if math.prod(shape):
            required += 1 + sum(
                (size - 1) * step
                for size, step in zip(shape, stride, strict=True)
            )
        raw = storages[record["storage"]]
        if required * target.element_size() > raw.numel():
            raise ValueError(f"Tensor exceeds backing storage: {name}")
        signature = (
            record["storage"],
            offset,
            tuple(shape),
            tuple(stride),
            record["dtype"],
        )
        if id(target) in objects and objects[id(target)] != signature:
            raise ValueError(f"Conflicting tied tensor alias: {name}")
        objects[id(target)] = signature
        byte_offset = raw.storage_offset() + offset * target.element_size()
        if byte_offset % target.element_size():
            raise ValueError(f"Unaligned tensor storage: {name}")
        view = torch.empty(0, dtype=target.dtype, device=target_device).set_(
            raw.untyped_storage(),
            byte_offset // target.element_size(),
            tuple(shape),
            tuple(stride),
        )
        if _byte_digest(view) != record["sha256"]:
            raise ValueError(f"Tensor digest mismatch: {name}")
        views[name] = view
    # Validate non-persistent buffers before mutating the model as well.
    ephemeral = []
    persistent_objects = {id(target) for target, _ in bindings.values()}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name not in bindings and id(buffer) not in persistent_objects:
            ephemeral.append((buffer, buffer.to(device=target_device)))
    for name, (target, _) in bindings.items():
        target.data = views[name]
    for target, value in ephemeral:
        target.data = value
    model._sllm_storage_owners = storages
