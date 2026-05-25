#include "flat.hpp"
#include "genivf.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "types.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <print>
#include <unordered_map>

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();

const std::string ref_seq = "data/dengue_ref_sequences.fasta";
const std::string index_file = "out.ivf.givf";
const std::string index_metadata_file = "out.ivf.meta.tsv";
const std::string fastq_file = "data/left.fq";
const std::string query_file = "query_file.tsv";

void
query_index(const std::string& fastq_file,
            const std::string& index_file,
            const std::string& index_metadata_file,
            const std::string& query_meta_file,
            size_t nprobe,
            size_t k = 1,
            MetricType metric = MetricType::HAMMING)
{
    std::vector<std::vector<std::uint8_t>> query_vectors;
    build_query_kmer_vectors<KMER_K, KMER_DIM>(fastq_file, query_vectors);

    // load ivf
    auto loaded_index = io::load_index(index_file);

    std::vector<Point> query_points;

    for (size_t i = 0; i < query_vectors.size(); ++i) {
        query_points.emplace_back(i, query_vectors[i]);
    }

    std::unordered_map<size_t, WindowMetaData> window_map;

    auto is_metadata_map_built =
      build_metadata_map_from_tsv(index_metadata_file, window_map);

    if (!is_metadata_map_built) {
        std::println(std::cerr, "Error creating metadata map");
        return;
    }

    std::ofstream tsv_file(query_meta_file);

    for (auto& point : query_points) {

        auto t_start = std::chrono::steady_clock::now();
        auto search_results = loaded_index.search(point, k, nprobe, metric);

        auto t_end = std::chrono::steady_clock::now();
        double elapsed =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "Query time: " << elapsed << "ms.\n";

        for (const auto& result : search_results) {
            const auto& meta = window_map.at(result.id);
            // FIX: add the query sequence header too.
            tsv_file << point.id << "\t"
                     << meta.sequence_name << "\t"
                     << meta.start_pos << "\t" << result.distance << "\n";
        }
    }
}

int
main()
{
    genivf::log::set_level(genivf::log::Level::INFO);
    size_t nprobe = 16;
    query_index(
      fastq_file, index_file, index_metadata_file, query_file, nprobe);
    return 0;
}
