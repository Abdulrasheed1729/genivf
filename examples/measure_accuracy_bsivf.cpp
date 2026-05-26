#include "bsivf.hpp"
#include "flat.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <array>
#include <print>
#include <vector>

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();
using KmerVector = std::array<std::uint8_t, KMER_DIM>;

static double
compute_recall(const std::string& fasta_file,
               const std::string& fastq_file,
               const std::string& flat_index_file,
               size_t stride,
               size_t k,
               genivf::MetricType metric)
{
    size_t total_expected = 0;
    size_t total_found = 0;

    std::vector<WindowMetaData> windows;
    std::vector<KmerVector> kmer_vectors;

    build_kmer_vectors_from_fasta_file<KMER_K, KMER_DIM>(
      fasta_file, kmer_vectors, windows);

    size_t num_of_bytes = KMER_DIM >> 3;

    std::vector<uint8_t> packed_vectors;
    auto is_packed_vectors_built =
      build_packed_kmer_vectors(kmer_vectors, packed_vectors);
    if (!is_packed_vectors_built)
        return 0.0;

    std::vector<Point> data_points;

    for (size_t i = 0; i < kmer_vectors.size(); ++i) {
        std::vector<uint8_t> v(packed_vectors.data() + i * num_of_bytes,
                               packed_vectors.data() + (i + 1) * num_of_bytes);
        data_points.emplace_back(i, std::move(v));
    }

    IndexBSIVF bsivf(data_points[0].values.size(), data_points.size(), stride);

    bsivf.construct_centroids();

    bsivf.add(data_points);

    auto flat = io::load_flat_index(flat_index_file);

    std::vector<std::vector<std::uint8_t>> query_vectors;
    build_query_kmer_vectors<KMER_K, KMER_DIM>(fastq_file, query_vectors);

    std::vector<Point> query_points;

    for (size_t i = 0; i < query_vectors.size(); ++i) {
        query_points.emplace_back(i, query_vectors[i]);
    }

    for (const auto& query : query_points) {
        auto ground_truth = flat.search(query, k, 1, metric);
        auto approx = bsivf.search(query, metric);

        std::vector<size_t> gt_ids;
        gt_ids.reserve(ground_truth.size());
        for (const auto& r : ground_truth) {
            gt_ids.push_back(r.id);
        }

        bool found = false;
        for (size_t id : gt_ids) {
            if (approx.id == id) {
                found = true;
                break;
            }
        }

        total_expected += 1;
        total_found += found;
    }

    return static_cast<double>(total_found) /
           static_cast<double>(total_expected);
}

int
main()
{
    genivf::log::set_level(genivf::log::Level::NONE);

    const std::string fasta_file = "data/dengue_ref_sequences.fasta";
    const std::string fastq_file = "data/left.fq";
    const std::string flat_index_file = "out.flat.givf";
    size_t stride = 25;
    size_t k = 1;
    double recall = compute_recall(
      fasta_file, fastq_file, flat_index_file, stride, k, MetricType::HAMMING);

    std::print("  {:.4f}", recall);

    return 0;
}
