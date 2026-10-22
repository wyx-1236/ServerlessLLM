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
#include "tensor_writer.h"

#include <iostream>
#include <stdexcept>

TensorWriter::TensorWriter(const std::string& filename) : filename_(filename) {}

TensorWriter::~TensorWriter() {}

uint64_t TensorWriter::writeRecord(const char* data, size_t size) {
  if (failure_) {
    std::rethrow_exception(failure_);
  }
  if (finished_) {
    throw std::logic_error("Tensor writer is already finished");
  }
  try {
    if (partition_idx_ == -1 || partition_size_ + size > kPartitionMaxSize) {
      if (buffer_) {
        buffer_->finish();
      }
      // Publish the partition state only after its buffer is constructed.
      const int next_partition = partition_idx_ + 1;
      std::string partition_filename =
          filename_ + "_" + std::to_string(next_partition);
      buffer_ = std::make_unique<AlignedBuffer>(partition_filename);
      partition_idx_ = next_partition;
      partition_size_ = 0;
    }

    uint64_t start_offset = offset_;
    // make sure the data is 64-bit aligned
    size_t padding = (size % 8) ? (8 - size % 8) : 0;
    size_t written = buffer_->writeData(data, size);
    if (padding) {
      written += buffer_->writePadding(padding);
    }
    offset_ += written;
    partition_size_ += written;

    return start_offset;
  } catch (...) {
    failure_ = std::current_exception();
    throw;
  }
}

void TensorWriter::finish() {
  if (failure_) {
    std::rethrow_exception(failure_);
  }
  try {
    if (buffer_) {
      buffer_->finish();
    }
    finished_ = true;
  } catch (...) {
    failure_ = std::current_exception();
    throw;
  }
}
