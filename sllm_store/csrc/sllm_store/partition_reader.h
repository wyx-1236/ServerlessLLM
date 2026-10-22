// Copyright (c) ServerlessLLM Team 2024
// Licensed under the Apache License, Version 2.0.
// See the repository LICENSE file for license terms.
#pragma once

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace sllm_store {

// Files retain their logical length. O_DIRECT requests must still have aligned
// addresses, offsets and lengths, including the short final filesystem block.
inline void ReadFileRange(int fd, char* destination, size_t offset,
                          size_t size) {
  constexpr size_t alignment = 4096;
  alignas(alignment) std::array<char, 64 * 1024> scratch;
  while (size != 0) {
    const size_t aligned_offset = offset - offset % alignment;
    const size_t prefix = offset - aligned_offset;
    const bool direct =
        prefix == 0 &&
        reinterpret_cast<uintptr_t>(destination) % alignment == 0 &&
        size >= alignment;
    const size_t maximum =
        static_cast<size_t>(std::numeric_limits<ssize_t>::max());
    const size_t request = direct
                               ? std::min(size, maximum) / alignment * alignment
                               : scratch.size();
    if (aligned_offset >
        static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      throw std::out_of_range("Partition read offset exceeds off_t");
    }
    void* buffer = direct ? static_cast<void*>(destination) : scratch.data();
    ssize_t received;
    do {
      received = pread(fd, buffer, request, static_cast<off_t>(aligned_offset));
    } while (received < 0 && errno == EINTR);
    if (received < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "Partition pread failed");
    }
    if (static_cast<size_t>(received) <= prefix) {
      throw std::runtime_error("Unexpected end of Store partition");
    }
    const size_t copied =
        std::min(size, static_cast<size_t>(received) - prefix);
    if (!direct) {
      std::memcpy(destination, scratch.data() + prefix, copied);
    }
    destination += copied;
    offset += copied;
    size -= copied;
  }
}

// Read a logical range from any number of physical partitions. The caller owns
// descriptors and publishes a chunk only after this entire operation succeeds.
inline void ReadPartitionRange(const std::vector<int>& descriptors,
                               const std::vector<size_t>& sizes,
                               char* destination, size_t offset, size_t size) {
  if (descriptors.size() != sizes.size() || (size != 0 && !destination)) {
    throw std::invalid_argument("Invalid Store partition read");
  }
  size_t partition = 0;
  while (partition < sizes.size() && offset >= sizes[partition]) {
    offset -= sizes[partition++];
  }
  while (size != 0) {
    if (partition == sizes.size()) {
      throw std::out_of_range("Read exceeds Store partitions");
    }
    const size_t copied = std::min(size, sizes[partition] - offset);
    ReadFileRange(descriptors[partition], destination, offset, copied);
    destination += copied;
    size -= copied;
    offset = 0;
    ++partition;
  }
}

}  // namespace sllm_store
