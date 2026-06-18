#include "bsivf.hpp"
#include "flat.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <print>
#include <vector>

using namespace genivf;
using namespace genivf::log;
constexpr size_t KMER_K = 5;
constexpr size_t KMER_DIM = kmer_vector_size<KMER_K>();
using KmerVector = std::array<std::uint8_t, KMER_DIM>;

struct SearchMetrics
{
    // Recall
    double recall = 0.0;

    // Timing — full pipeline
    double total_build_time_ms = 0.0;
    double total_search_time_ms = 0.0;
    double total_gt_time_ms = 0.0;

    // Per-query stats
    double mean_search_time_us = 0.0;
    double median_search_time_us = 0.0;
    double p95_search_time_us = 0.0;
    double p99_search_time_us = 0.0;
    double min_search_time_us = 0.0;
    double max_search_time_us = 0.0;
    double stddev_search_time_us = 0.0;

    // Throughput
    double queries_per_second = 0.0;
    double gt_queries_per_second = 0.0;

    // Distance stats
    double mean_approx_distance = 0.0;
    double mean_gt_distance = 0.0;
    double mean_distance_margin = 0.0; // approx - gt, lower is better

    // Index info
    size_t num_data_points = 0;
    size_t num_query_points = 0;
    size_t num_centroids = 0;
    size_t total_found = 0;
    size_t total_expected = 0;

    void print() const
    {
        log::info("====== Search Benchmark Results ======");
        log::info("  Index size:          {} points", num_data_points);
        log::info("  Num centroids:       {}", num_centroids);
        log::info("  Num queries:         {}", num_query_points);
        log::info("");
        log::info("  -- Recall --");
        log::info("  Recall:              {:.4f} ({}/{})",
                  recall,
                  total_found,
                  total_expected);
        log::info("");
        log::info("  -- Build times --");
        log::info("  Index build time:    {:.2f} ms", total_build_time_ms);
        log::info("  GT search time:      {:.2f} ms", total_gt_time_ms);
        log::info("  BSIVF search time:   {:.2f} ms", total_search_time_ms);
        log::info("");
        log::info("  -- Per-query latency (BSIVF) --");
        log::info("  Mean:                {:.2f} us", mean_search_time_us);
        log::info("  Median:              {:.2f} us", median_search_time_us);
        log::info("  P95:                 {:.2f} us", p95_search_time_us);
        log::info("  P99:                 {:.2f} us", p99_search_time_us);
        log::info("  Min:                 {:.2f} us", min_search_time_us);
        log::info("  Max:                 {:.2f} us", max_search_time_us);
        log::info("  Stddev:              {:.2f} us", stddev_search_time_us);
        log::info("");
        log::info("  -- Throughput --");
        log::info("  BSIVF QPS:           {:.1f}", queries_per_second);
        log::info("  GT QPS:              {:.1f}", gt_queries_per_second);
        log::info("  Speedup vs GT:       {:.2f}x",
                  gt_queries_per_second > 0
                    ? queries_per_second / gt_queries_per_second
                    : 0.0);
        log::info("");
        log::info("  -- Distance quality --");
        log::info("  Mean approx dist:    {:.4f}", mean_approx_distance);
        log::info("  Mean GT dist:        {:.4f}", mean_gt_distance);
        log::info("  Mean margin:         {:.4f}", mean_distance_margin);
        log::info("======================================");
    }
};

static std::pair<double, SearchMetrics>
compute_recall(const std::string& fasta_file,
               const std::string& fastq_file,
               const std::string& flat_index_file,
               const std::string& eval_file_path,
               size_t stride,
               size_t min_stride,
               size_t k,
               size_t nprobe,
               size_t tolerance,
               genivf::MetricType metric)
{
    SearchMetrics metrics{};

    // ---- Build index ----
    auto build_start = std::chrono::high_resolution_clock::now();

    std::vector<WindowMetaData> windows;
    std::vector<KmerVector> kmer_vectors;
    build_kmer_vectors_from_fasta_file<KMER_K, KMER_DIM>(
      fasta_file, kmer_vectors, windows);

    size_t num_of_bytes = KMER_DIM >> 3;
    std::vector<uint8_t> packed_vectors;
    auto is_packed_vectors_built =
      build_packed_kmer_vectors(kmer_vectors, packed_vectors);
    if (!is_packed_vectors_built)
        return { 0.0, metrics };

    std::vector<Point> data_points;
    data_points.reserve(kmer_vectors.size());
    for (size_t i = 0; i < kmer_vectors.size(); ++i) {
        std::vector<uint8_t> v(packed_vectors.data() + i * num_of_bytes,
                               packed_vectors.data() + (i + 1) * num_of_bytes);
        data_points.emplace_back(i, std::move(v));
    }

    IndexBSIVF bsivf(data_points[0].values.size(), data_points.size());
    bsivf.construct_centroids(stride);
    bsivf.add(data_points);

    auto build_end = std::chrono::high_resolution_clock::now();
    metrics.total_build_time_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start)
        .count();
    metrics.num_data_points = data_points.size();
    metrics.num_centroids = bsivf.num_centroids();

    // ---- Build query points ----
    auto flat = io::load_flat_index(flat_index_file);
    std::vector<std::vector<std::uint8_t>> query_vectors;
    build_query_kmer_vectors<KMER_K, KMER_DIM>(fastq_file, query_vectors);

    std::vector<Point> query_points;
    query_points.reserve(query_vectors.size());
    for (size_t i = 0; i < query_vectors.size(); ++i) {
        query_points.emplace_back(i, query_vectors[i]);
    }
    metrics.num_query_points = query_points.size();

    // ---- Search loop ----
    std::ofstream eval_file(eval_file_path);
    eval_file << "query_id,approx_id,approx_dist,rank,"
                 "gt_id,gt_dist,margin,"
                 "search_time_us,gt_time_us\n";

    size_t total_expected = 0;
    size_t total_found = 0;

    std::vector<double> search_times_us;
    std::vector<double> gt_times_us;
    search_times_us.reserve(query_points.size());
    gt_times_us.reserve(query_points.size());

    double sum_approx_dist = 0.0;
    double sum_gt_dist = 0.0;
    double sum_margin = 0.0;

    for (const auto& query : query_points) {
        // Time ground truth search
        auto gt_start = std::chrono::high_resolution_clock::now();
        auto ground_truth = flat.search(query, k, metric);
        auto gt_end = std::chrono::high_resolution_clock::now();
        double gt_us =
          std::chrono::duration<double, std::micro>(gt_end - gt_start).count();
        gt_times_us.push_back(gt_us);

        // Time BSIVF search
        auto search_start = std::chrono::high_resolution_clock::now();
        auto approx = bsivf.search(query, stride, min_stride, nprobe, metric);
        auto search_end = std::chrono::high_resolution_clock::now();
        double search_us =
          std::chrono::duration<double, std::micro>(search_end - search_start)
            .count();
        search_times_us.push_back(search_us);

        bool found = false;
        for (size_t rank = 0; rank < ground_truth.size(); ++rank) {
            const auto& gt = ground_truth[rank];
            double margin = approx.distance - gt.distance;

            sum_approx_dist += approx.distance;
            sum_gt_dist += gt.distance;
            sum_margin += margin;

            eval_file << query.id << "," << approx.id << "," << approx.distance
                      << "," << rank << "," << gt.id << "," << gt.distance
                      << "," << margin << "," << search_us << "," << gt_us
                      << "\n";

            double offset = std::abs(static_cast<double>(approx.id) -
                                     static_cast<double>(gt.id));
            if (offset <= static_cast<double>(tolerance)) {
                found = true;
            }
        }

        total_expected += 1;
        total_found += found;
    }

    // ---- Aggregate timing stats ----
    auto total_search_us =
      std::accumulate(search_times_us.begin(), search_times_us.end(), 0.0);
    auto total_gt_us =
      std::accumulate(gt_times_us.begin(), gt_times_us.end(), 0.0);

    metrics.total_search_time_ms = total_search_us / 1000.0;
    metrics.total_gt_time_ms = total_gt_us / 1000.0;

    // Per-query latency distribution
    std::sort(search_times_us.begin(), search_times_us.end());
    size_t n = search_times_us.size();

    metrics.min_search_time_us = search_times_us.front();
    metrics.max_search_time_us = search_times_us.back();
    metrics.mean_search_time_us = total_search_us / static_cast<double>(n);
    metrics.median_search_time_us = search_times_us[n / 2];
    metrics.p95_search_time_us = search_times_us[static_cast<size_t>(n * 0.95)];
    metrics.p99_search_time_us = search_times_us[static_cast<size_t>(n * 0.99)];

    double variance = 0.0;
    for (double t : search_times_us) {
        double diff = t - metrics.mean_search_time_us;
        variance += diff * diff;
    }
    metrics.stddev_search_time_us =
      std::sqrt(variance / static_cast<double>(n));

    // Throughput
    metrics.queries_per_second = n / (total_search_us / 1e6);
    metrics.gt_queries_per_second = n / (total_gt_us / 1e6);

    // Distance quality
    double total_results = static_cast<double>(total_expected * k);
    metrics.mean_approx_distance = sum_approx_dist / total_results;
    metrics.mean_gt_distance = sum_gt_dist / total_results;
    metrics.mean_distance_margin = sum_margin / total_results;

    // Recall
    metrics.recall =
      static_cast<double>(total_found) / static_cast<double>(total_expected);
    metrics.total_found = total_found;
    metrics.total_expected = total_expected;

    metrics.print();

    return { metrics.recall, metrics };
}

int
main()
{
    genivf::log::set_level(genivf::log::Level::NONE);

    const std::string fasta_file = "data/dengue_ref_sequences.fasta";
    const std::string fastq_file = "data/left.fq";
    const std::string flat_index_file = "out.flat.givf";

    const size_t k = 1;
    const size_t nprobe = 25;

    const std::array<size_t, 3> strides = { 10, 25, 50 };
    const std::array<size_t, 5> min_strides = { 1, 2, 4, 8, 16 };

    const std::string summary_file = "bsivf_summary_nprobe_tol_by_stride.csv";
    std::ofstream summary(summary_file);

    // CSV header
    summary << "stride,min_stride,"
               "recall,"
               "build_time_ms,"
               "total_search_time_ms,total_gt_time_ms,"
               "mean_search_us,median_search_us,"
               "p95_search_us,p99_search_us,"
               "min_search_us,max_search_us,stddev_search_us,"
               "qps,gt_qps,speedup,"
               "mean_approx_dist,mean_gt_dist,mean_margin,"
               "num_data_points,num_query_points,num_centroids\n";

    // Console header
    std::print("{:>6} {:>10} {:>8} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} "
               "{:>10} {:>12}\n",
               "stride",
               "min_str",
               "recall",
               "build_ms",
               "search_ms",
               "gt_ms",
               "mean_us",
               "p99_us",
               "qps",
               "gt_qps",
               "speedup");
    std::print("{}\n", std::string(114, '-'));

    for (auto stride : strides) {
        for (auto min_stride : min_strides) {
            if (min_stride >= stride)
                continue;

            std::string eval_file =
              std::format("bsivf_s{}_ms{}.eval.csv", stride, min_stride);

            auto [recall, m] = compute_recall(fasta_file,
                                              fastq_file,
                                              flat_index_file,
                                              eval_file,
                                              stride,
                                              min_stride,
                                              k,
                                              nprobe,
                                              stride,
                                              MetricType::HAMMING);

            // Write full metrics row to CSV
            summary << stride << "," << min_stride << "," << m.recall << ","
                    << m.total_build_time_ms << "," << m.total_search_time_ms
                    << "," << m.total_gt_time_ms << "," << m.mean_search_time_us
                    << "," << m.median_search_time_us << ","
                    << m.p95_search_time_us << "," << m.p99_search_time_us
                    << "," << m.min_search_time_us << ","
                    << m.max_search_time_us << "," << m.stddev_search_time_us
                    << "," << m.queries_per_second << ","
                    << m.gt_queries_per_second << ","
                    << (m.gt_queries_per_second > 0
                          ? m.queries_per_second / m.gt_queries_per_second
                          : 0.0)
                    << "," << m.mean_approx_distance << ","
                    << m.mean_gt_distance << "," << m.mean_distance_margin
                    << "," << m.num_data_points << "," << m.num_query_points
                    << "," << m.num_centroids << "\n";

            // Flush per row so partial results are visible on long runs
            summary.flush();

            // Console — key metrics only
            double speedup = m.gt_queries_per_second > 0
                               ? m.queries_per_second / m.gt_queries_per_second
                               : 0.0;
            std::print("{:>6} {:>10} {:>8.4f} {:>12.2f} {:>12.2f} {:>12.2f} "
                       "{:>10.2f} {:>10.2f} {:>10.1f} {:>10.1f} {:>11.2f}x\n",
                       stride,
                       min_stride,
                       m.recall,
                       m.total_build_time_ms,
                       m.total_search_time_ms,
                       m.total_gt_time_ms,
                       m.mean_search_time_us,
                       m.p99_search_time_us,
                       m.queries_per_second,
                       m.gt_queries_per_second,
                       speedup);
        }

        // Separator between stride groups for readability
        std::print("{}\n", std::string(114, '-'));
    }

    std::print("\nSummary written to {}\n", summary_file);
    return 0;
}
