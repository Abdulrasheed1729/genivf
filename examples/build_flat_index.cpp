#include "flat.hpp"
#include "io.hpp"
#include "logger.hpp"

#include <print>
#include <vector>

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();

const std::string ref_seq = "data/dengue_ref_sequences.fasta";
const std::string flat_index_file = "out.flat.givf";
const std::string flat_index_metadata_file = "out.flat.meta.tsv";

using KmerVector = std::array<uint8_t, KMER_DIM>;

std::unique_ptr<IndexFlat>
build_flat_index(const std::string& fasta_file,
                 const std::string& index_output_file,
                 const std::string& index_metadata_file)
{
    // build kmer vectors
    std::vector<WindowMetaData> windows;
    std::vector<KmerVector> kmer_vectors;

    build_kmer_vectors_from_fasta_file<KMER_K, KMER_DIM>(
      fasta_file, kmer_vectors, windows);

    size_t num_of_bytes = KMER_DIM >> 3;

    std::vector<uint8_t> packed_vectors;
    auto is_packed_vectors_built =
      build_packed_kmer_vectors(kmer_vectors, packed_vectors);
    if (!is_packed_vectors_built)
        return nullptr;

    std::vector<Point> data_points;

    for (size_t i = 0; i < kmer_vectors.size(); ++i) {
        std::vector<uint8_t> v(packed_vectors.data() + i * num_of_bytes,
                               packed_vectors.data() + (i + 1) * num_of_bytes);
        data_points.emplace_back(i, std::move(v));
    }

    IndexFlat flat(data_points[0].values.size(), data_points.size());

    flat.add(data_points);

    io::save_flat_index(flat, index_output_file);

    auto is_metadata_built = build_metadata_file(index_metadata_file, windows);

    if (!is_metadata_built)
        return nullptr;

    return std::make_unique<IndexFlat>(std::move(flat));
}
int
main()
{

    genivf::log::set_level(genivf::log::Level::INFO);

    auto flat_index =
      build_flat_index(ref_seq, flat_index_file, flat_index_metadata_file);

    if (!flat_index) {
        std::println(std::cerr, "Error building flat index");
        return 1;
    }

    return 0;
}
