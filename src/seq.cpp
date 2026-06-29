#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "seq.hpp"

namespace genivf::seq {

FastaScanner::FastaScanner(const std::string& filename)
  : filename(filename)
  , io_buffer(128 * 1024) // 128KB buffer to minimize syscall overhead
{
    file.rdbuf()->pubsetbuf(io_buffer.data(), io_buffer.size());
    file.open(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + filename);
    }
}

bool
FastaScanner::hasNext() const
{
    return has_pending_header || !reached_eof;
}

FastaRecord
FastaScanner::next()
{
    FastaRecord record;
    std::string line;

    if (has_pending_header) {
        record.header = std::move(pending_header);
        has_pending_header = false;
    } else {
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }

            if (line[0] == '>') {
                record.header = line.substr(1);
                break;
            }

            throw std::runtime_error("Invalid FASTA: sequence before header");
        }
    }

    if (record.header.empty()) {
        reached_eof = true;
        return record;
    }

    // Pre-reserve dynamic allocation space to eliminate constant reallocation
    record.sequence.reserve(4096);

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '>') {
            pending_header = line.substr(1);
            has_pending_header = true;
            return record;
        }

        // Fast path: if the line has no spaces (which is typical), append it
        // directly.
        bool has_space = false;
        for (char c : line) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                has_space = true;
                break;
            }
        }

        if (!has_space) {
            if (record.sequence.size() + line.size() >
                record.sequence.capacity()) {
                record.sequence.reserve(
                  std::max(record.sequence.capacity() * 2,
                           record.sequence.size() + line.size() + 4096));
            }
            record.sequence.append(line);
        } else {
            // Slow path fallback: skip whitespace
            for (char c : line) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    if (record.sequence.size() >= record.sequence.capacity()) {
                        record.sequence.reserve(record.sequence.capacity() * 2 +
                                                4096);
                    }
                    record.sequence.push_back(c);
                }
            }
        }
    }

    reached_eof = true;
    return record;
}

FastqScanner::FastqScanner(const std::string& filename)
  : filename(filename)
  , io_buffer(128 * 1024) // 128KB buffer to minimize syscall overhead
{
    file.rdbuf()->pubsetbuf(io_buffer.data(), io_buffer.size());
    file.open(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + filename);
    }
}

bool
FastqScanner::hasNext()
{
    return file.peek() != EOF;
}

FastqRecord
FastqScanner::next()
{
    FastqRecord record;
    std::string line;

    // Line 1: header (starts with '@')
    if (!std::getline(file, line) || line.empty() || line[0] != '@') {
        throw std::runtime_error(
          "Invalid FASTQ: expected header line starting with '@'");
    }
    record.header = line.substr(1);

    // Line 2: sequence
    if (!std::getline(file, line)) {
        throw std::runtime_error("Invalid FASTQ: expected sequence line");
    }
    record.sequence = std::move(line);

    // Line 3: '+' separator
    if (!std::getline(file, line) || line.empty() || line[0] != '+') {
        throw std::runtime_error("Invalid FASTQ: expected '+' separator line");
    }

    // Line 4: quality scores
    if (!std::getline(file, line)) {
        throw std::runtime_error("Invalid FASTQ: expected quality line");
    }
    record.quality = std::move(line);

    if (record.quality.length() != record.sequence.length()) {
        throw std::runtime_error("Invalid FASTQ: quality length (" +
                                 std::to_string(record.quality.length()) +
                                 ") does not match sequence length (" +
                                 std::to_string(record.sequence.length()) +
                                 ") for record: " + record.header);
    }

    return record;
}
} // namespace genivf::seq
