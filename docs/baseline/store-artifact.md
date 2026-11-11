# Store artifact transaction and tensor layout

This specifies the stage-B artifact contract. The disk integrity and logical
view helpers do not themselves qualify a vLLM loader or a model. Publication
requires the preparation pipeline's independent restore verification; native
Store ABI, GPU capacity and engine correctness require separate acceptance.

## Disk layout

```text
model/
  config.json, tokenizer files, ...
  rank_0/
    tensor.data_0, tensor.data_1, ...
    tensor_index.json
    tensor-layout.json
  rank_1/ ...                    # exactly world_size ranks
  artifact-manifest.json
  artifact-commit.json
```

The existing Store byte partitions and five-field tensor index remain the
transport format. The versioned layout uses index keys `storage_0`, etc., for
contiguous uint8 backing-storage vectors. Logical model names live in the
sidecar; they are not directly loadable with the historical model-name loader.
This representation retains shared storage, transposed views and nonzero
storage offsets without assuming that every tensor is contiguous or that tied
weights can be discarded. Each backing storage is exported once.

`tensor-layout.json` has `schema_version: 1`, a `storages` mapping containing
`nbytes` and SHA256, and a `tensors` mapping containing kind (`parameter` or
`buffer`), storage key, storage_offset (in elements of that tensor's dtype),
shape, stride, dtype and logical tensor SHA256. Scalar and empty tensor shapes
are explicit. Digests cover bytes in logical contiguous tensor order; backing
storage digests cover all storage bytes. Persistent buffers are included.
Non-persistent buffers are re-created by the model and transferred without
changing dtype. Non-tensor state and unsupported layouts fail explicitly.

Restore checks every key, kind, shape, stride, dtype, storage bound and digest
before assigning any persistent state. Conflicting views of the same tied
Parameter object fail. The model retains the raw Store tensors as
`_sllm_storage_owners`: the raw tensors own the native allocation, while logical
views must not outlive that allocation. A complete backend shutdown still owns
engine/Worker process cleanup; this reference is not a shutdown mechanism.

## Identity and publication

`artifact-manifest.json` schema 1 contains an exact JSON `identity`, a complete
file inventory with sizes/SHA256, and the successful independent restore
verification receipt. Identity includes model source/revision, engine/version,
dtype, TP world_size and writer; the preparation pipeline also supplies plugin,
layout and effective export configuration identities. Engine layout or TP
changes require a new artifact.

`artifact-commit.json` schema 1 records the exact manifest file SHA256. It is
created only after the verifier returns a successful receipt. A caller must not
substitute a byte-only or mock verifier for native/Store/engine acceptance.

`ArtifactTransaction` creates a unique sibling staging directory on the target
filesystem. It validates contiguous ranks/partitions and tensor-index bounds,
rejects symlinks, hashes the complete payload, invokes the restore verifier,
checks that payload bytes did not change during verification, writes the
manifest and marker, flushes files/directories, then renames the directory.
Cooperating publishers serialize the final existence check and rename with a
short-lived directory lock. Linux publication also uses `renameat2` with
`RENAME_NOREPLACE`, so even a non-cooperating concurrent empty-directory creator
cannot be replaced; unsupported filesystems fail closed. This is a filesystem publication lock, not a GPU
reservation. A foreign/stale lock causes failure and is never automatically
removed. Existing targets are never replaced by the transaction.

Failure removes only that transaction's staging directory. Reuse requires
`validate_artifact(path, expected_identity)`: marker, manifest digest, exact
identity, rank/index bounds and the complete file inventory are checked again.
A pre-existing directory, an old artifact without a marker, changed files or
unexpected payload files cannot be treated as prepared. Legacy adoption must
run an explicit conversion/verification path; these helpers do not infer it.

## Native writer completion

The writer retains O_DIRECT and 8-byte logical record alignment. A partial disk
block is zero-filled to the 4096-byte disk boundary, written with checked I/O,
then truncated to its logical length. Record padding is zero. Existing output
files are truncated when opened. Aligned short writes and EINTR are handled;
other failures propagate and remain terminal. `SaveTensors` explicitly finishes
the writer before returning offsets, so Python cannot publish an index after
an unnoticed destructor write failure. Destructor cleanup is best effort and
does not substitute for explicit successful completion.

The Store host reader consumes logical partition lengths rather than assuming
that every file ends at a disk block boundary. Aligned bulk reads retain the
direct-to-host-buffer path. Unaligned heads, tails or destinations use a bounded
aligned scratch buffer and copy only logical bytes. Read requests keep 4096-byte
alignment even when the final file read returns fewer bytes at EOF. Logical
ranges may span more than two partitions; each chunk is published only after
its complete range has been read. EINTR retries, while premature EOF and I/O
errors fail the chunk. CPU reader tests do not qualify CUDA IPC or engine restore.

## Preparation entry and ownership

### Public Store export commands

`sllm-store save --backend vllm` and
`python sllm_store/examples/save_vllm_model.py` delegate through
`sllm_store.vllm_export.export_vllm_model` to the same preparation pipeline.
The preparation supervisor and Worker are owned by `sllm_store.vllm_preparation`
and `sllm_store.vllm_prepare_worker`. Store export uses these modules directly.
Standalone use requires the Store vLLM extra and the pinned Linux/native runtime.
The original `sllm.vllm_preparation` and `sllm.vllm_prepare_worker` import paths
forward to the same modules, including the old Worker `python -m` entry.

Wheels and source distributions include `preparation-lock.sha256`, generated
from the repository's `uv.lock` by the Store build commands. Rebuilding a wheel
from a standalone source distribution preserves its bundled digest. Installed
preparation reads that digest; source-checkout use reads the repository lock
when no bundled digest exists. Invalid bundled digests fail validation. The
artifact identity still uses `dependency_lock_sha256` with the same full-lock
SHA256 meaning. This provenance records the build's declared environment and
does not by itself prove the installed dependencies or native ABI match it.

The command names, options, downloader method signatures and successful `None`
return are preserved. Their existing path contracts differ.

| Entry | Storage path priority | Artifact destination |
| --- | --- | --- |
| Store CLI | Explicit `--storage-path`, then nonempty `STORAGE_PATH`, then `~/models/` | `<root>/vllm/<model>` |
| Export example | `STORAGE_PATH` when set, then `--storage-path`, then `~/models`; an empty resolved value falls back to `~/models` | `<root>/<model>` |

The public downloader methods accept `pattern` and `max_size` and report that
they are ignored, matching the original patch's behavior. These are Python
method parameters, not additional command-line options.

With `--local-model-path`, the directory is checked before preparation and its
contents supply the frozen source identity. With only a model name, the adapter
resolves the Hub default branch to a full commit before preparation and passes
that revision to the download/verification chain. Resolution errors propagate
without starting preparation. The existing command therefore needs no new
required revision argument.

Preparation's optional `flat_layout=True` writes `<storage_path>/<model>` for
these legacy export roots. Other callers retain the default
`<storage_path>/vllm/<model>` layout. Both layouts use the same staging,
independent offline verification, publication and reuse validation. Failure
cleanup belongs to the transaction, preserving pre-existing target data.

These commands launch real export and verification workers when used normally
and require GPU approval. CPU entry-point tests replace the Hub and preparation
calls. Pipeline tests retain real source files and transactions while replacing
child processes. Those checks do not establish real Hub, native Store, engine
or native ABI qualification for these commands. Python-only packaging tests
build source distributions and wheels with native compilation replaced, install
the wheels without dependencies in fresh environments, and block main-package
imports while checking the export and Worker entry points.

### Main package preparation

The main package's `VllmModelDownloader.download_vllm_model` also accepts the
legacy `pattern` and `max_size` arguments, including positional calls. Supplying
either argument with a non-`None` value logs that both are ignored. An empty
pattern or zero size retains the same behavior. Preparation receives the full
model and the supplied source/revision, dtype and TP identity; these legacy
arguments do not filter tensors or change partition sizes. Successful calls
return the preparation result and preparation failures propagate to the caller.

`VllmModelDownloader.download_vllm_model` delegates to
`sllm.vllm_preparation.prepare_vllm_model`. Hub sources require a fixed lowercase
40-character commit; local sources use an actual complete content inventory and
`sha256:<inventory digest>` revision. The source repo is the supplied pretrained
source, while the model name is only the Store alias. Optional `source_snapshot`
is a receipt mapping or JSON path returned by `freeze_source`; its bytes and
identity are rechecked before launching any child. Freeze/download sources
before a GPU batch so network/download time does not consume its approval.

Preparation is Linux-only and bounded by an overall deadline of at most 1800
seconds. It first probes a new user/network namespace using `unshare`; failure
prevents export. A native-load child generates the frozen raw prompts and
exports each actual TP rank. After that child and its owned process group exit,
a second child restores only the candidate artifact in a separate network
namespace. The namespace has only loopback; its private Store uses an ephemeral
port and does not register with the shared service. Prompt token IDs, generated
tokens and top-5 logprobs must match. Store data and logical tensor/storage
digests are rechecked. Both child processes have an owner-death watchdog.

The StoreManager waits for successful committed preparation before registering
rank paths, and kills its temporary downloader actor on success, failure or
cancellation. Invalid rank-size responses cannot publish success metadata.
Native registration has no rollback RPC: a partial rank registration may remain
in the Store, but the model is not marked complete and retry is allowed. The
loader uses the private endpoint only during verification; serving integration
and resource lifetime qualification are owned by later stages.

Engine commands may initialize CUDA and require explicit server/GPU approval.
Process-group/watchdog tests cannot prove recovery of processes that escape the
owned session or actual CUDA/Ray resources. Real server namespace availability,
native extension ABI, memory peaks, independent restore and cleanup remain
qualification requirements. This contract does not claim Stage B acceptance.
