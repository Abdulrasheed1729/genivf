#include "ivf.hpp"
#include "logger.hpp"

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();

const std::string ref_seq = "data/dengue_ref_sequences.fasta";
const std::string index_file = "out.ivf.givf";
const std::string index_metadata_file = "out.ivf.meta.tsv";

using KmerVector = std::array<uint8_t, KMER_DIM>;

std::unique_ptr<IndexIVF>
build_flat_index(const std::string& fasta_file,
                 const std::string& index_output_file,
                 const std::string& index_metadata_file,
                 const size_t nlist)
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

    IndexIVF ivf(nlist, num_of_bytes);

    ivf.train(data_points);
    ivf.add(data_points);

    io::save_index(ivf, index_output_file);

    auto is_metadata_built = build_metadata_file(index_metadata_file, windows);

    if (!is_metadata_built)
        return nullptr;

    return std::make_unique<IndexIVF>(std::move(ivf));
}

int
main()
{

    genivf::log::set_level(genivf::log::Level::INFO);

    size_t nlist = 256;

    std::unique_ptr<IndexIVF> ivf =
      build_flat_index(ref_seq, index_file, index_metadata_file, nlist);

    if (!ivf)
        return 1;

    return 0;
}
