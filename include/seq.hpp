#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace genivf::seq {

// Represents a FASTA record (header and sequence).
struct FastaRecord
{
    std::string header;
    std::string sequence;
};

// Represents a FASTQ record (header, sequence, and quality).
struct FastqRecord
{
    std::string header;
    std::string sequence;
    std::string quality;
};

// Scans a FASTA file and yields FastaRecord objects.
struct FastaScanner
{

  public:
    explicit FastaScanner(const std::string& filename);
    [[nodiscard]] bool hasNext() const;
    [[nodiscard]] FastaRecord next();

    ~FastaScanner() { file.close(); };

  private:
    std::string filename;
    std::vector<char> io_buffer;
    std::ifstream file;
    std::string pending_header;
    bool has_pending_header = false;
    bool reached_eof = false;
};

// Scans a FASTQ file and yields FastqRecord objects.
struct FastqScanner
{

  public:
    explicit FastqScanner(const std::string& filename);
    ~FastqScanner() { file.close(); };
    [[nodiscard]] bool hasNext();
    [[nodiscard]] FastqRecord next();

  private:
    std::string filename;
    std::vector<char> io_buffer;
    std::ifstream file;
};
}

