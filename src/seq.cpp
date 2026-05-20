#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../include/seq.hpp"

namespace genivf::seq {
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
        record.header = pending_header;
        has_pending_header = false;
    } else {
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

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

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        if (line[0] == '>') {
            pending_header = line.substr(1);
            has_pending_header = true;
            return record;
        }

        for (char c : line) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                record.sequence.push_back(c);
            }
        }
    }

    reached_eof = true;
    return record;
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
    record.sequence = line;

    // Line 3: '+' separator
    if (!std::getline(file, line) || line.empty() || line[0] != '+') {
        throw std::runtime_error("Invalid FASTQ: expected '+' separator line");
    }

    // Line 4: quality scores
    if (!std::getline(file, line)) {
        throw std::runtime_error("Invalid FASTQ: expected quality line");
    }
    record.quality = line;

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
