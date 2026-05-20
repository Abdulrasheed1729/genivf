#include <fstream>
#include <stdexcept>
#include <string>

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
    explicit FastaScanner(const std::string& filename)
      : filename(filename)
      , file(filename)
    {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }
    };
    bool hasNext() const;
    FastaRecord next();

    ~FastaScanner() { file.close(); };

  private:
    std::string filename;
    std::ifstream file;
    std::string pending_header;
    bool has_pending_header = false;
    bool reached_eof = false;
};

// Scans a FASTQ file and yields FastqRecord objects.
struct FastqScanner
{

  public:
    explicit FastqScanner(const std::string& filename)
      : filename(filename)
      , file(filename)
    {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }
    };
    ~FastqScanner() { file.close(); };
    bool hasNext();
    FastqRecord next();

  private:
    std::string filename;
    std::ifstream file;
};
}
