#include "bsivf.hpp"
#include "flat.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <array>
#include <cstdint>
#include <fstream>
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
               const std::string& eval_file_path,
               size_t stride,
               size_t min_stride,
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

    IndexBSIVF bsivf(data_points[0].values.size(), data_points.size());

    bsivf.construct_centroids(stride);

    bsivf.add(data_points);

    auto flat = io::load_flat_index(flat_index_file);

    std::vector<std::vector<std::uint8_t>> query_vectors;
    build_query_kmer_vectors<KMER_K, KMER_DIM>(fastq_file, query_vectors);

    std::vector<Point> query_points;

    for (size_t i = 0; i < query_vectors.size(); ++i) {
        query_points.emplace_back(i, query_vectors[i]);
    }

    std::ofstream eval_file(eval_file_path);

    eval_file << "query_id,approx_id,approx_dist,rank,gt_id,gt_dist,margin\n";

    for (const auto& query : query_points) {
        auto ground_truth = flat.search(query, k, metric);
        auto approx = bsivf.search(query, stride, min_stride, metric);

        bool found = false;
        for (size_t rank = 0; rank < ground_truth.size(); ++rank) {
            const auto& gt = ground_truth[rank];
            double margin = approx.distance - gt.distance;
            eval_file << query.id << "," << approx.id << "," << approx.distance
                      << "," << rank << "," << gt.id << "," << gt.distance
                      << "," << margin << "\n";
            if (approx.id == gt.id) {
                found = true;
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
    const size_t k = 1;

    const std::array<size_t, 4> strides = { 10, 25, 50, 100 };
    const std::array<size_t, 5> min_strides = { 1, 2, 4, 8, 16 };

    const std::string summary_file = "bsivf_summary.csv";
    std::ofstream summary(summary_file);
    summary << "stride,min_stride,recall\n";

    std::print("{:>6} {:>10} {:>8}\n", "stride", "min_stride", "recall");

    for (auto stride : strides) {
        for (auto min_stride : min_strides) {
            if (min_stride >= stride)
                continue;

            std::string eval_file =
              std::format("bsivf_s{}_ms{}.eval.csv", stride, min_stride);

            double recall = compute_recall(fasta_file,
                                           fastq_file,
                                           flat_index_file,
                                           eval_file,
                                           stride,
                                           min_stride,
                                           k,
                                           MetricType::HAMMING);

            summary << stride << "," << min_stride << "," << recall << "\n";
            std::print("{:>6} {:>10} {:>7.4f}\n", stride, min_stride, recall);
        }
    }
    std::print("\nSummary written to {}\n", summary_file);

    return 0;
}
