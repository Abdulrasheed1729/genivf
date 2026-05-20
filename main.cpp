#include "genivf.hpp"
#include "io.hpp"
#include "seq.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <strings.h>
#include <vector>

using namespace genivf;
using namespace genivf::seq;

using KmerVector = std::vector<uint8_t>;

constexpr size_t KMER_K = 5;

struct WindowMetaData
{
    std::shared_ptr<const std::string> sequence_name;
    int start_pos = 0;
};
template<size_t KMER>
static constexpr size_t
kmer_vector_size()
{
    return (size_t(1) << (2u * KMER));
}

// Branchless lookup table — faster than switch, GPU-friendly (no branches)
// Maps ASCII -> 2-bit encoding: A=0, C=1, T=2, G=3, invalid=-1
static constexpr int8_t BASE_LUT[256] = {
    // Generated: only A/a=0, C/c=1, T/t=2, G/g=3 are valid
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, 0,  -1, 1,  -1, -1, -1, 3,  -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 0,  -1, 1,  -1, -1, -1, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

inline int8_t
base_to_bits(char base)
{
    return BASE_LUT[static_cast<uint8_t>(base)];
}

template<size_t KMER>
bool
kmer_one_hot_raw(const char* sequence, size_t seq_len, uint8_t* out)
{
    constexpr size_t N = kmer_vector_size<KMER>();
    constexpr size_t mask = N - 1;

    std::memset(out, 0, N);

    if (seq_len < KMER)
        return false; // invalid sequence length

    // Seed the first k-mer
    size_t index = 0;
    for (size_t i = 0; i < KMER; ++i) {
        int8_t code = base_to_bits(sequence[i]);
        if (code == -1) {
            // *error_flag = 1;
            // return false;
            // HACK: this is just for the code to run if there is a non-ACTG
            // character in the sequence
            continue;
        }
        index = (index << 2) | static_cast<size_t>(code);
    }
    out[index] = 1;

    // Slide the window
    for (size_t i = KMER; i < seq_len; ++i) {
        int8_t code = base_to_bits(sequence[i]);
        if (code == -1) {
            // *error_flag = 1;
            // return false;
            // HACK: this is just for the code to run if there is a non-ACTG
            // character in the sequence
            continue;
        }
        index = ((index << 2) & mask) | static_cast<size_t>(code);
        out[index] = 1;
    }

    return true;
}

template<size_t KMER>
void
kmer_one_hot_batch(
  const char* const* sequences, // array of C-strings
  const size_t* seq_lens,       // length of each sequence
  size_t num_seqs,
  uint8_t* out // pre-allocated: num_seqs * kmer_vector_size<KMER>()
)
{
    constexpr size_t N = kmer_vector_size<KMER>();
    for (size_t s = 0; s < num_seqs; ++s) {
        kmer_one_hot_raw<KMER>(sequences[s], seq_lens[s], out + s * N);
    }
}

template<size_t KMER>
std::vector<uint8_t>
kmer_one_hot(const std::string& sequence)
{
    constexpr size_t N = kmer_vector_size<KMER>();
    std::vector<uint8_t> result(N, 0);

    kmer_one_hot_raw<KMER>(sequence.data(), sequence.size(), result.data());

    return result;
}

inline void
pack_kmer_one_hot(const std::vector<uint8_t>& vec_in,
                  std::vector<uint8_t>& vec_out)
{
    const size_t num_bits = vec_in.size();
    const size_t byte_size = (vec_in.size() + 7) >> 3;
    vec_out.resize(byte_size);

    for (size_t byte_idx = 0; byte_idx < byte_size; ++byte_idx) {
        uint8_t current_byte = 0;
        const size_t start_bit = byte_idx * 8;
        const size_t limit = std::min(start_bit + 8, num_bits);

        for (size_t i = start_bit; i < limit; ++i) {
            if (vec_in[i]) {
                current_byte |=
                  static_cast<std::uint8_t>(1u << (7 - (i - start_bit)));
            }
        }

        vec_out[byte_idx] = current_byte;
    }
}

std::tuple<std::vector<KmerVector>, std::vector<WindowMetaData>>
process_fasta_file(const std::string& fasta_path,
                   size_t window_size = 50,
                   size_t stride = 1)
{
    // load fasta file and return a string of sequence

    FastaScanner scanner(fasta_path);

    // NOTE: for now we are only assuming that we have a single sequence fasta
    // file
    FastaRecord record = scanner.next();
    std::string sequence = record.sequence;

    std::vector<KmerVector> sequences;
    std::vector<WindowMetaData> window_metas;

    // Wrap sequence name in a shared pointer to avoid thousands of string
    // copies
    auto seq_name_ptr =
      std::make_shared<const std::string>(std::move(record.header));

    // make a sliding window run of each overlapping strings and encode each
    // window as a vector of uint8_t

    for (size_t i = 0; i + window_size <= sequence.size(); i += stride) {
        constexpr size_t N = kmer_vector_size<KMER_K>();
        KmerVector kmer_vector(N, 0);
        // Avoid substring allocations by encoding directly from sequence view
        // slice
        kmer_one_hot_raw<KMER_K>(
          sequence.data() + i, window_size, kmer_vector.data());

        sequences.push_back(std::move(kmer_vector));
        WindowMetaData meta;
        meta.sequence_name = seq_name_ptr;
        meta.start_pos = i;
        window_metas.push_back(std::move(meta));
    }

    return std::make_tuple(std::move(sequences), std::move(window_metas));
}

bool
build_ivf_index(const std::string& index_path,
                const std::string& data_path, // size_t d,
                size_t nlist)
{

    // assert(d % 8 == 0); // d must be greater than 0

    auto d_bits = kmer_vector_size<5>();

    auto [vectors, windows] = process_fasta_file(data_path);

    assert(!vectors.empty());

    auto d_bytes = d_bits >> 3;
    IndexIVF index(nlist, d_bytes);

    auto nb = vectors.size();
    std::vector<Point> points;

    if (nb <= 0)
        return false;

    std::vector<uint8_t> packed_buffer;
    for (size_t i = 0; i < vectors.size(); ++i) {
        pack_kmer_one_hot(vectors[i], packed_buffer);
        points.push_back(Point(i, packed_buffer));
    }

    if (!index.is_trained() && nb > nlist) {
        std::cout << "Training the Binary IVF index ...\n";
        index.train(points, 100);
    }

    index.add(points);
    io::save_index(index, index_path);

    std::cout << "Index saved to " << index_path << "\n";

    std::cout << "Saving metadata ...\n";

    std::ofstream meta_file(index_path + ".meta.ivf.tsv");
    for (size_t i = 0; i < windows.size(); ++i) {
        meta_file << i << "\t"
                  << (windows[i].sequence_name ? *windows[i].sequence_name : "")
                  << "\t" << windows[i].start_pos << "\n";
    }

    std::cout << "Metadata saved to " << index_path << ".meta.ivf.tsv\n";

    return true;
}

std::vector<SearchResult>
query_index(const IndexIVF& loaded_index,
            const std::unordered_map<size_t, WindowMetaData>& meta_map,
            const FastqRecord& query_sequence,
            int k,
            size_t nprobe,
            size_t query_id,
            std::ofstream& tsv_file)
{
    KmerVector query_vec_unpacked =
      kmer_one_hot<KMER_K>(query_sequence.sequence);
    std::vector<uint8_t> query_vec;
    pack_kmer_one_hot(query_vec_unpacked, query_vec);
    Point query_point(query_id, query_vec);

    auto t_start = std::chrono::steady_clock::now();
    auto results =
      loaded_index.search(query_point, k, nprobe, MetricType::HAMMING);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Query time: " << elapsed << "ms.\n";
    for (int i = 0; i < k; ++i) {
        const auto& meta = meta_map.at(results[i].id);
        std::cout << (meta.sequence_name ? *meta.sequence_name : "") << ":"
                  << meta.start_pos << " (distance: " << results[i].distance
                  << ") " << query_sequence.header << "\n";

        // Collect result in TSV file
        tsv_file << query_id << "\t" << query_sequence.header << "\t"
                 << (meta.sequence_name ? *meta.sequence_name : "") << "\t"
                 << meta.start_pos << "\t" << results[i].distance << "\n";
    }

    return results;
}

int
main()
{
    genivf::log::set_level(genivf::log::Level::INFO);

    std::string ref_seq = "data/dengue_ref_sequences.fasta";
    std::string left_seq = "data/right.fq";

    auto is_ivf_index_built = build_ivf_index("main.givf", ref_seq, 256);
    if (!is_ivf_index_built) {
        std::println("Failed to build IVF index.");
        return 1;
    }

    // Load index and metadata once to eliminate I/O bottleneck in search loop
    std::cout << "Loading index and metadata for batch querying...\n";
    auto loaded_index = io::load_index("main.givf");

    std::unordered_map<size_t, WindowMetaData> meta_map;
    std::ifstream meta_file("main.givf.meta.ivf.tsv");
    std::string line;
    while (std::getline(meta_file, line)) {
        size_t idx;
        std::string idx_str;
        std::string start_pos_str;
        std::string seq_name;
        WindowMetaData meta;
        std::istringstream iss(line);
        std::getline(iss, idx_str, '\t');
        std::getline(iss, seq_name, '\t');
        std::getline(iss, start_pos_str, '\t');
        idx = std::stoul(idx_str.c_str());
        meta.start_pos = std::stoi(start_pos_str);
        meta.sequence_name =
          std::make_shared<const std::string>(std::move(seq_name));
        meta_map[idx] = std::move(meta);
    }

    // Open TSV file for collecting results
    std::ofstream tsv_file("query_results.tsv");
    tsv_file << "query_id\tquery_header\ttarget_sequence\ttarget_start_"
                "pos\tdistance\n";

    FastqScanner scanner(left_seq);

    size_t k = 1;
    size_t nprobe = 32;
    size_t count = 0;

    auto t_start = std::chrono::steady_clock::now();

    while (scanner.hasNext()) {
        auto record = scanner.next();
        auto results = query_index(
          loaded_index, meta_map, record, k, nprobe, count, tsv_file);
        count++;
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_elapsed =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\nBatch querying completed successfully!\n";
    std::cout << "Total queries processed: " << count << "\n";
    std::cout << "Total querying execution time: " << total_elapsed << " ms.\n";
    std::cout << "Results collected into: query_results.tsv\n";
}
