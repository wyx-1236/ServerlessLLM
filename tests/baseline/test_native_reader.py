"""Exercise the production partition reader with real Linux O_DIRECT files."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sllm_store/csrc/sllm_store"
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("g++") is None,
    reason="requires Linux and g++; no GPU required",
)


@pytest.fixture(scope="module")
def reader_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("native-reader")
    harness = build / "reader.cpp"
    harness.write_text(
        r"""
#include "partition_reader.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

std::string fault;
int calls = 0;
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* data, size_t count, off_t offset) {
  if (++calls == 1) {
    if (fault == "eintr") { errno = EINTR; return -1; }
    if (fault == "eio") { errno = EIO; return -1; }
    if (fault == "zero") return 0;
    if (fault == "short") {
      const ssize_t result = __real_pread(fd, data, count, offset);
      return result > 7 ? 7 : result;
    }
  }
  return __real_pread(fd, data, count, offset);
}

int main(int argc, char** argv) {
  const size_t offset = std::stoull(argv[1]);
  const size_t count = std::stoull(argv[2]);
  const size_t shift = std::stoull(argv[3]);
  fault = argv[4];
  std::vector<int> descriptors;
  std::vector<size_t> sizes;
  std::vector<char> expected;
  for (int i = 5; i < argc; ++i) {
    const int fd = open(argv[i], O_RDONLY | O_DIRECT);
    if (fd < 0) return 20;
    descriptors.push_back(fd);
    struct stat info;
    if (fstat(fd, &info) != 0) return 21;
    sizes.push_back(info.st_size + (fault == "truncated" ? 4096 : 0));
    const int buffered = open(argv[i], O_RDONLY);
    std::vector<char> bytes(info.st_size);
    if (read(buffered, bytes.data(), bytes.size()) != info.st_size) return 22;
    close(buffered);
    expected.insert(expected.end(), bytes.begin(), bytes.end());
  }
  void* allocation = nullptr;
  if (posix_memalign(&allocation, 4096, count + shift + 8192) != 0) return 23;
  auto* data = static_cast<char*>(allocation);
  std::memset(data, 123, count + shift + 8192);
  bool rejected = false;
  try {
    sllm_store::ReadPartitionRange(descriptors, sizes, data + 4096 + shift,
                                  offset, count);
  } catch (const std::system_error& error) {
    if (fault != "eio" || error.code().value() != EIO) return 24;
    rejected = true;
  } catch (const std::exception&) {
    if (fault != "zero" && fault != "truncated" && fault != "range") return 25;
    rejected = true;
  }
  const bool expect_error = fault == "eio" || fault == "zero" ||
                            fault == "truncated" || fault == "range";
  if (rejected != expect_error) return 26;
  if (!rejected && count && std::memcmp(data + 4096 + shift, expected.data() + offset,
                               count) != 0) return 27;
  for (size_t i = 0; i < 4096 + shift; ++i) if (data[i] != 123) return 28;
  for (size_t i = 4096 + shift + count; i < count + shift + 8192; ++i)
    if (data[i] != 123) return 29;
  free(allocation);
  for (int fd : descriptors) close(fd);
  std::cout << "calls=" << calls << "\n";
}
""",
        encoding="utf-8",
    )
    binary = build / "reader"
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
            "-Wl,--wrap=pread",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def read_range(binary, tmp_path, sizes, offset, count, shift=0, fault="none"):
    files = []
    position = 0
    for index, size in enumerate(sizes):
        path = tmp_path / f"partition-{index}"
        path.write_bytes(bytes((position + i) % 251 for i in range(size)))
        position += size
        files.append(str(path))
    return subprocess.run(
        [str(binary), str(offset), str(count), str(shift), fault, *files],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 8193, 65537])
@pytest.mark.parametrize("shift", [0, 1])
def test_direct_read_logical_tail(reader_binary, tmp_path, size, shift):
    read_range(reader_binary, tmp_path, [size], 0, size, shift)


@pytest.mark.parametrize(
    "sizes,offset,count",
    [
        ([4095, 4097, 1], 0, 8193),
        ([1, 0, 3, 8193], 1, 8196),
        ([65537, 4095, 7], 4095, 65544),
        ([4096, 4096], 4096, 4096),
        ([8193], 4095, 4098),
    ],
)
def test_direct_read_partition_boundaries(
    reader_binary, tmp_path, sizes, offset, count
):
    read_range(reader_binary, tmp_path, sizes, offset, count, shift=7)


@pytest.mark.parametrize("fault", ["eintr", "short", "eio", "zero"])
def test_direct_read_io_outcomes(reader_binary, tmp_path, fault):
    read_range(reader_binary, tmp_path, [8193], 0, 8193, fault=fault)


def test_truncated_partition_rejected(reader_binary, tmp_path):
    read_range(reader_binary, tmp_path, [4097], 0, 8193, fault="truncated")


def test_out_of_range_rejected(reader_binary, tmp_path):
    read_range(reader_binary, tmp_path, [4096], 4096, 1, fault="range")
