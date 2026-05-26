#include "flat.hpp"
#include "ivf.hpp"
#include "logger.hpp"
#include "types.hpp"

#include <algorithm>
#include <print>
#include <vector>

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();

static double
compute_recall(const std::string& fastq_file,
               const std::string& flat_index_file,
               const std::string& ivf_index_file,
               const size_t k,
               const size_t nprobe,
               const MetricType metric)
{
    size_t total_expected = 0;
    size_t total_found = 0;

    auto ivf = io::load_index(ivf_index_file);
    auto flat = io::load_flat_index(flat_index_file);

    std::vector<std::vector<std::uint8_t>> query_vectors;
    build_query_kmer_vectors<KMER_K, KMER_DIM>(fastq_file, query_vectors);

    std::vector<Point> query_points;

    for (size_t i = 0; i < query_vectors.size(); ++i) {
        query_points.emplace_back(i, query_vectors[i]);
    }

    for (const auto& query : query_points) {
        auto ground_truth = flat.search(query, k, 1, metric);
        auto approx = ivf.search(query, k, nprobe, metric);

        std::vector<size_t> gt_ids;
        gt_ids.reserve(ground_truth.size());
        for (const auto& r : ground_truth) {
            gt_ids.push_back(r.id);
        }

        std::vector<size_t> approx_ids;
        approx_ids.reserve(approx.size());
        for (const auto& r : approx) {
            approx_ids.push_back(r.id);
        }
        std::ranges::sort(approx_ids);

        size_t found = 0;
        for (size_t id : gt_ids) {
            if (std::ranges::binary_search(approx_ids, id)) {
                ++found;
            }
        }

        total_expected += ground_truth.size();
        total_found += found;
    }

    return static_cast<double>(total_found) /
           static_cast<double>(total_expected);
}

int
main()
{
    genivf::log::set_level(genivf::log::Level::NONE);

    const std::string fastq_file = "data/left.fq";
    const std::string flat_index_file = "out.flat.givf";
    const std::string ivf_index_file = "out.ivf.givf";
    double recall = compute_recall(
      fastq_file, flat_index_file, ivf_index_file, 1, 16, MetricType::HAMMING);

    std::print("  {:.4f}", recall);

    return 0;
}
