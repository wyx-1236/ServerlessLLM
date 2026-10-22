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
