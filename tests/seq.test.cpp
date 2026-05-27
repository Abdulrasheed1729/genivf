#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "seq.hpp"
#include "doctest.h"

#include <filesystem>
#include <fstream>
#include <string>

TEST_CASE("seq: FastaScanner basic parsing")
{
    std::string test_fasta = "test_seq.fasta";
    {
        std::ofstream out(test_fasta);
        out << ">seq1\n";
        out << "ACGT\n";
        out << "TGCA\n";
        out << "\n"; // empty line
        out << ">seq2 second sequence\n";
        out << "AAAA\n";
    }

    genivf::seq::FastaScanner scanner(test_fasta);

    CHECK(scanner.hasNext());
    auto r1 = scanner.next();
    CHECK_EQ(r1.header, "seq1");
    CHECK_EQ(r1.sequence, "ACGTTGCA");

    CHECK(scanner.hasNext());
    auto r2 = scanner.next();
    CHECK_EQ(r2.header, "seq2 second sequence");
    CHECK_EQ(r2.sequence, "AAAA");

    CHECK_FALSE(scanner.hasNext());

    std::filesystem::remove(test_fasta);
}

TEST_CASE("seq: FastqScanner basic parsing")
{
    std::string test_fastq = "test_seq.fastq";
    {
        std::ofstream out(test_fastq);
        out << "@read1\n";
        out << "ACGT\n";
        out << "+\n";
        out << "IIII\n";
        out << "@read2 second\n";
        out << "AAAA\n";
        out << "+\n";
        out << "####\n";
    }

    genivf::seq::FastqScanner scanner(test_fastq);

    CHECK(scanner.hasNext());
    auto r1 = scanner.next();
    CHECK_EQ(r1.header, "read1");
    CHECK_EQ(r1.sequence, "ACGT");
    CHECK_EQ(r1.quality, "IIII");

    CHECK(scanner.hasNext());
    auto r2 = scanner.next();
    CHECK_EQ(r2.header, "read2 second");
    CHECK_EQ(r2.sequence, "AAAA");
    CHECK_EQ(r2.quality, "####");

    CHECK_FALSE(scanner.hasNext());

    std::filesystem::remove(test_fastq);
}

TEST_CASE("seq: FastqScanner quality sequence mismatch error")
{
    std::string test_fastq = "test_mismatch.fastq";
    {
        std::ofstream out(test_fastq);
        out << "@read1\n";
        out << "ACGT\n";
        out << "+\n";
        out << "III\n"; // 3 chars vs 4 bases
    }

    genivf::seq::FastqScanner scanner(test_fastq);
    CHECK(scanner.hasNext());
    CHECK_THROWS_AS((void)scanner.next(), std::runtime_error);

    std::filesystem::remove(test_fastq);
}
