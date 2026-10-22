"""Compile production disk writers alone, including real O_DIRECT fault tests."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sllm_store/csrc/checkpoint"
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None,
    reason="requires Linux and a C++ compiler; no GPU required",
)


@pytest.fixture(scope="module")
def writer_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("native-writer")
    harness = build / "writer.cpp"
    harness.write_text(
        r"""
#include "tensor_writer.h"
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

std::string fault;
int write_calls = 0, truncate_calls = 0, sync_calls = 0;
int open_calls = 0, allocation_calls = 0;
int observed_fd = -1;
extern "C" int __real_open(const char*, int, ...);
extern "C" void* __real_aligned_alloc(size_t, size_t);
extern "C" int __wrap_open(const char* path, int flags, ...) {
  ++open_calls;
  if (open_calls == 1 && fault == "open") { errno = EACCES; return -1; }
  va_list args;
  va_start(args, flags);
  const mode_t mode = va_arg(args, mode_t);
  va_end(args);
  observed_fd = __real_open(path, flags, mode);
  return observed_fd;
}
extern "C" void* __wrap_aligned_alloc(size_t alignment, size_t size) {
  ++allocation_calls;
  if (allocation_calls == 1 && fault == "allocate") return nullptr;
  return __real_aligned_alloc(alignment, size);
}
extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" int __real_ftruncate(int, off_t);
extern "C" int __real_fsync(int);
extern "C" ssize_t __wrap_pwrite(int fd, const void* data, size_t size,
                                  off_t offset) {
  ++write_calls;
  observed_fd = fd;
  if (write_calls == 1) {
    if (fault == "eintr") { errno = EINTR; return -1; }
    if (fault == "enospc") { errno = ENOSPC; return -1; }
    if (fault == "zero") return 0;
    if (fault == "unaligned") return 1;
    if (fault == "short" && size > 4096) size = 4096;
  }
  // Keep actual writes and O_DIRECT enabled, including the short-write suffix.
  return __real_pwrite(fd, data, size, offset);
}
extern "C" int __wrap_ftruncate(int fd, off_t size) {
  ++truncate_calls;
  observed_fd = fd;
  if (truncate_calls == 1 && fault == "truncate") {
    errno = EIO; return -1;
  }
  return __real_ftruncate(fd, size);
}
extern "C" int __wrap_fsync(int fd) {
  ++sync_calls;
  observed_fd = fd;
  if (sync_calls == 1 && fault == "sync") { errno = EIO; return -1; }
  return __real_fsync(fd);
}

int main(int, char** argv) {
  const std::string mode = argv[2];
  const size_t count = std::stoull(argv[3]);
  const size_t buffer_size = std::stoull(argv[4]);
  fault = argv[5];
  std::vector<char> data(count);
  for (size_t i = 0; i < count; ++i) data[i] = char(i % 251);
  if (mode == "constructor-error") {
    TensorWriter writer(argv[1]);
    try { writer.writeRecord(data.data(), data.size()); return 37; }
    catch (const std::exception& first) {
      const std::exception* original = &first;
      if (fault == "open") {
        const auto* error = dynamic_cast<const std::system_error*>(&first);
        if (!error || error->code().value() != EACCES) return 38;
      } else {
        if (!dynamic_cast<const std::bad_alloc*>(&first)) return 39;
        if (fcntl(observed_fd, F_GETFD) != -1 || errno != EBADF) return 40;
      }
      try { writer.finish(); return 41; }
      catch (const std::exception& repeated) {
        if (&repeated != original) return 42;
      }
      try { writer.writeRecord("x", 1); return 43; }
      catch (const std::exception& repeated) {
        if (&repeated != original) return 44;
      }
      if (open_calls != 1 || allocation_calls != (fault == "allocate")) return 45;
      std::cout << first.what() << "\n";
    }
  } else if (mode == "writer-error") {
    TensorWriter writer(argv[1]);
    writer.writeRecord(data.data(), data.size());
    try { writer.finish(); return 29; }
    catch (const std::system_error& first) {
      const std::string message = first.what();
      try { writer.finish(); return 30; }
      catch (const std::system_error& repeated) {
        if (message != repeated.what()) return 31;
      }
      try { writer.writeRecord("x", 1); return 32; }
      catch (const std::system_error&) {}
      std::cout << message << "\n";
    }
  } else if (mode == "invalid-buffer") {
    try { AlignedBuffer writer(argv[1], buffer_size); return 33; }
    catch (const std::invalid_argument&) { return 0; }
  } else if (mode == "records" || mode == "destructor") {
    TensorWriter writer(argv[1]);
    std::cout << writer.writeRecord(data.data(), data.size()) << " ";
    std::cout << writer.writeRecord("abc", 3) << "\n";
    if (mode == "destructor") return 0;
    writer.finish();
    writer.finish();
    try { writer.writeRecord("x", 1); return 20; }
    catch (const std::logic_error&) {}
    try { writer.finish(); }
    catch (...) { return 46; }
  } else if (mode == "empty-finish") {
    TensorWriter writer(argv[1]);
    writer.finish();
    writer.finish();
    try { writer.writeRecord("x", 1); return 21; }
    catch (const std::logic_error&) {}
    try { writer.finish(); }
    catch (...) { return 47; }
  } else {
    AlignedBuffer writer(argv[1], buffer_size);
    const bool expect_error = fault == "enospc" || fault == "zero" ||
        fault == "unaligned" || fault == "truncate" || fault == "sync";
    try {
      writer.writeData(data.data(), data.size());
      writer.finish();
      if (expect_error) return 22;
    } catch (const std::exception& first) {
      if (!expect_error) throw;
      const auto* system_error = dynamic_cast<const std::system_error*>(&first);
      if (fault == "enospc" &&
          (!system_error || system_error->code().value() != ENOSPC)) return 34;
      if ((fault == "truncate" || fault == "sync") &&
          (!system_error || system_error->code().value() != EIO)) return 35;
      if (fcntl(observed_fd, F_GETFD) != -1 || errno != EBADF) return 36;
      const std::string message = first.what();
      const int calls = write_calls + truncate_calls + sync_calls;
      try { writer.finish(); return 23; }
      catch (const std::exception& repeated) {
        if (message != repeated.what()) return 24;
      }
      try { writer.writeData("x", 1); return 25; }
      catch (const std::exception&) {}
      if (calls != write_calls + truncate_calls + sync_calls) return 26;
      std::cout << message << "\n";
      return 0;
    }
    const int calls = write_calls + truncate_calls + sync_calls;
    writer.finish();
    if (calls != write_calls + truncate_calls + sync_calls) return 27;
    try { writer.writeData("x", 1); return 28; }
    catch (const std::logic_error&) {}
  }
  std::cout << "writes=" << write_calls << "\n";
}
"""
    )
    binary = build / "writer"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(SOURCE),
            str(harness),
            str(SOURCE / "tensor_writer.cpp"),
            str(SOURCE / "aligned_buffer.cpp"),
            "-Wl,--wrap=pwrite",
            "-Wl,--wrap=ftruncate",
            "-Wl,--wrap=fsync",
            "-o",
            str(binary),
            "-Wl,--wrap=open",
            "-Wl,--wrap=aligned_alloc",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def run_writer(
    binary, target, count, *, mode="buffer", buffer_size=8192, fault="none"
):
    return subprocess.run(
        [str(binary), str(target), mode, str(count), str(buffer_size), fault],
        check=True,
        capture_output=True,
        text=True,
    )


def payload(count):
    return bytes(i % 251 for i in range(count))


@pytest.mark.parametrize(
    "count", [0, 1, 7, 8, 4095, 4096, 4097, 8191, 8192, 8193, 24577]
)
@pytest.mark.parametrize("buffer_size", [4096, 8192])
def test_native_buffer_persists_exact_bytes(
    writer_binary, tmp_path, count, buffer_size
):
    target = tmp_path / "tensor.data"
    target.write_bytes(b"stale" * 10000)
    run_writer(writer_binary, target, count, buffer_size=buffer_size)
    assert target.read_bytes() == payload(count)


@pytest.mark.parametrize("count", [0, 1, 7, 8, 4095, 4096, 4097])
def test_native_writer_offsets_and_zero_padding(writer_binary, tmp_path, count):
    target = tmp_path / "tensor.data"
    result = run_writer(writer_binary, target, count, mode="records")
    first_size = count + (-count) % 8
    assert result.stdout.splitlines()[0] == f"0 {first_size}"
    assert target.with_name("tensor.data_0").read_bytes() == (
        payload(count) + bytes((-count) % 8) + b"abc" + bytes(5)
    )


@pytest.mark.parametrize("fault", ["short", "eintr"])
def test_native_buffer_retries_recoverable_writes(
    writer_binary, tmp_path, fault
):
    target = tmp_path / "tensor.data"
    result = run_writer(writer_binary, target, 20001, fault=fault)
    assert result.stdout.strip() == "writes=4"
    assert target.read_bytes() == payload(20001)


@pytest.mark.parametrize(
    "fault", ["zero", "unaligned", "enospc", "truncate", "sync"]
)
@pytest.mark.parametrize("count", [17, 8192, 20001])
def test_native_buffer_io_failure_is_terminal(
    writer_binary, tmp_path, fault, count
):
    result = run_writer(
        writer_binary, tmp_path / "tensor.data", count, fault=fault
    )
    assert result.stdout.strip()


def test_native_writer_empty_finish_is_terminal(writer_binary, tmp_path):
    target = tmp_path / "tensor.data"
    run_writer(writer_binary, target, 0, mode="empty-finish")
    assert not target.with_name("tensor.data_0").exists()


def test_native_writer_destructor_retains_best_effort_flush(
    writer_binary, tmp_path
):
    target = tmp_path / "tensor.data"
    run_writer(writer_binary, target, 4097, mode="destructor")
    assert target.with_name("tensor.data_0").read_bytes() == (
        payload(4097) + bytes(7) + b"abc" + bytes(5)
    )


@pytest.mark.parametrize("fault", ["enospc", "truncate", "sync"])
def test_native_writer_explicit_finish_propagates_failure(
    writer_binary, tmp_path, fault
):
    result = run_writer(
        writer_binary,
        tmp_path / "tensor.data",
        17,
        mode="writer-error",
        fault=fault,
    )
    assert result.stdout.strip()


@pytest.mark.parametrize("buffer_size", [0, 1, 4095, 4097])
def test_native_buffer_rejects_invalid_size_before_open(
    writer_binary, tmp_path, buffer_size
):
    target = tmp_path / "tensor.data"
    target.write_bytes(b"keep existing file")
    run_writer(
        writer_binary, target, 0, mode="invalid-buffer", buffer_size=buffer_size
    )
    assert target.read_bytes() == b"keep existing file"


@pytest.mark.parametrize("fault", ["open", "allocate"])
def test_native_writer_constructor_failure_is_terminal(
    writer_binary, tmp_path, fault
):
    result = run_writer(
        writer_binary,
        tmp_path / "tensor.data",
        17,
        mode="constructor-error",
        fault=fault,
    )
    assert result.stdout.strip()
