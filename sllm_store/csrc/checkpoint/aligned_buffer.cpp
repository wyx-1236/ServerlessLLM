// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "aligned_buffer.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <system_error>

AlignedBuffer::AlignedBuffer(const std::string& filename, size_t buffer_size)
    : fd_(-1),
      buf_size_(buffer_size),
      buf_pos_(0),
      file_offset_(0),
      buffer_(nullptr) {
  if (!buf_size_ || buf_size_ % kAlignment) {
    throw std::invalid_argument(
        "Buffer size must be a positive multiple of 4096");
  }
  fd_ = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd_ < 0) {
    throw std::system_error(errno, std::generic_category(), "open " + filename);
  }
  buffer_ = aligned_alloc(kAlignment, buf_size_);
  if (!buffer_) {
    close(fd_);
    fd_ = -1;
    throw std::bad_alloc();
  }
}

AlignedBuffer::~AlignedBuffer() {
  try {
    if (!failure_) {
      finish();
    }
  } catch (const std::exception& error) {
    // Explicit finish() is required by SaveTensors to propagate failures.
    std::cerr << "Failed to finish tensor file: " << error.what() << std::endl;
  }
  free(buffer_);
  if (fd_ >= 0) {
    close(fd_);
  }
}

void AlignedBuffer::writeAll(size_t size) {
  size_t written = 0;
  while (written < size) {
    ssize_t result = pwrite(fd_, static_cast<char*>(buffer_) + written,
                            size - written, file_offset_ + written);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(errno, std::generic_category(), "pwrite tensor");
    }
    // Continuing an unaligned short write would violate O_DIRECT constraints.
    if (result == 0 || static_cast<size_t>(result) % kAlignment) {
      throw std::runtime_error("Incomplete unaligned tensor write");
    }
    written += static_cast<size_t>(result);
  }
}

void AlignedBuffer::flush() {
  writeAll(buf_size_);
  file_offset_ += buf_size_;
  buf_pos_ = 0;
}

size_t AlignedBuffer::writeData(const void* data, size_t size) {
  if (failure_) {
    std::rethrow_exception(failure_);
  }
  if (fd_ < 0) {
    throw std::logic_error("Tensor file is already finished");
  }
  try {
    size_t written = 0;
    while (written < size) {
      size_t to_write = std::min(size - written, buf_size_ - buf_pos_);
      memcpy(static_cast<char*>(buffer_) + buf_pos_,
             static_cast<const char*>(data) + written, to_write);
      buf_pos_ += to_write;
      written += to_write;
      if (buf_pos_ == buf_size_) {
        flush();
      }
    }
    return written;
  } catch (...) {
    closeAfterFailure();
    throw;
  }
}

size_t AlignedBuffer::writePadding(size_t padding_size) {
  if (padding_size >= 8) {
    throw std::invalid_argument("Tensor padding must be less than 8 bytes");
  }
  const char zeros[8] = {};
  return writeData(zeros, padding_size);
}

void AlignedBuffer::finish() {
  if (failure_) {
    std::rethrow_exception(failure_);
  }
  if (fd_ < 0) {
    return;
  }
  try {
    const size_t logical_size = file_offset_ + buf_pos_;
    if (buf_pos_) {
      const size_t aligned_size =
          (buf_pos_ + kAlignment - 1) / kAlignment * kAlignment;
      memset(static_cast<char*>(buffer_) + buf_pos_, 0,
             aligned_size - buf_pos_);
      writeAll(aligned_size);
    }
    // Preserve Store's logical concatenated offsets; disk alignment is private.
    if (ftruncate(fd_, logical_size) < 0 || fsync(fd_) < 0) {
      throw std::system_error(errno, std::generic_category(), "finish tensor");
    }
    const int fd = fd_;
    fd_ = -1;
    if (close(fd) < 0) {
      throw std::system_error(errno, std::generic_category(), "close tensor");
    }
  } catch (...) {
    closeAfterFailure();
    throw;
  }
}

void AlignedBuffer::closeAfterFailure() {
  // A failed direct write may already have changed the file. Never retry it
  // or let a later finish falsely report this checkpoint as complete.
  failure_ = std::current_exception();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}
