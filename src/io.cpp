#include "io.hpp"

namespace genivf::io {

void
save_index(const IndexIVF& index, const std::filesystem::path& path)
{
    if (!index.is_trained()) {
        throw std::invalid_argument(
          "genivf::io::save_index: index must be trained before saving");
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error(
          "genivf::io::save_index: cannot open file for writing: " +
          path.string());
    }

    detail::write_bytes(ofs, kMagic, sizeof(kMagic));
    detail::write_val<uint8_t>(ofs, kVersion);
    detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(index.d_num_cells));
    detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(index.d_dim));
    detail::write_val<uint32_t>(ofs, static_cast<uint32_t>(index.d_seed));
    detail::write_val<uint64_t>(ofs,
                                static_cast<uint64_t>(index.d_vectors.size()));

    for (const Cluster& cluster : index.d_clusters) {
        detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(cluster.id));
        detail::write_val<uint64_t>(
          ofs, static_cast<uint64_t>(cluster.centroid.id));
        detail::write_bytes(ofs, cluster.centroid.values.data(), index.d_dim);

        detail::write_val<uint64_t>(
          ofs, static_cast<uint64_t>(cluster.point_indices.size()));
        for (const size_t pid : cluster.point_indices) {
            detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(pid));
        }
    }

    for (const auto& [id, point] : index.d_vectors) {
        detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(id));
        detail::write_bytes(ofs, point.values.data(), index.d_dim);
    }
}

IndexIVF
load_index(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("genivf::io::load_index: cannot open file: " +
                                 path.string());
    }

    uint8_t magic[4]{};
    detail::read_bytes(ifs, magic, sizeof(magic));
    if (magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
        magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
        throw std::runtime_error(
          "genivf::io::load_index: invalid magic — not a GIVF file");
    }

    const auto version = detail::read_val<uint8_t>(ifs);
    if (version != kVersion) {
        throw std::runtime_error(
          "genivf::io::load_index: unsupported file version " +
          std::to_string(version) + " (expected " + std::to_string(kVersion) +
          ")");
    }

    const auto num_cells =
      static_cast<size_t>(detail::read_val<uint64_t>(ifs));
    const auto dim = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
    const auto seed =
      static_cast<unsigned>(detail::read_val<uint32_t>(ifs));
    const auto num_vectors =
      static_cast<size_t>(detail::read_val<uint64_t>(ifs));

    IndexIVF index(num_cells, dim, seed);

    index.d_clusters.reserve(num_cells);
    for (size_t c = 0; c < num_cells; ++c) {
        const auto cluster_id =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        const auto centroid_id =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));

        std::vector<uint8_t> centroid_vals(dim);
        detail::read_bytes(ifs, centroid_vals.data(), dim);

        Cluster cluster(cluster_id,
                        Point(centroid_id, std::move(centroid_vals)));

        const auto list_len =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        cluster.point_indices.reserve(list_len);
        for (size_t i = 0; i < list_len; ++i) {
            cluster.point_indices.push_back(
              static_cast<size_t>(detail::read_val<uint64_t>(ifs)));
        }

        index.d_clusters.push_back(std::move(cluster));
    }

    if (index.d_clusters.size() != num_cells) {
        throw std::runtime_error(
          "genivf::io::load_index: cluster count mismatch in file");
    }

    index.d_vectors.reserve(num_vectors);
    for (size_t v = 0; v < num_vectors; ++v) {
        const auto pid = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        std::vector<uint8_t> vals(dim);
        detail::read_bytes(ifs, vals.data(), dim);
        index.d_vectors.emplace(pid, Point(pid, std::move(vals)));
    }

    if (index.d_vectors.size() != num_vectors) {
        throw std::runtime_error(
          "genivf::io::load_index: vector count mismatch in file");
    }

    for (auto& cluster : index.d_clusters) {
        cluster.flat_vectors.resize(cluster.point_indices.size() * dim);
        for (size_t i = 0; i < cluster.point_indices.size(); ++i) {
            const size_t pid = cluster.point_indices[i];
            auto it = index.d_vectors.find(pid);
            if (it == index.d_vectors.end()) {
                throw std::runtime_error(
                  "genivf::io::load_index: referential integrity violation — "
                  "cluster references non-existent point ID " +
                  std::to_string(pid));
            }
            std::ranges::copy(it->second.values,
                      &cluster.flat_vectors[i * dim]);
        }
    }

    return index;
}

void
save_flat_index(const IndexFlat& index, const std::filesystem::path& path)
{
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error(
          "genivf::io::save_flat_index: cannot open file for writing: " +
          path.string());
    }

    detail::write_bytes(ofs, kMagic, sizeof(kMagic));
    detail::write_val<uint8_t>(ofs, 2);
    detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(index.d_dim));
    detail::write_val<uint64_t>(ofs,
                                static_cast<uint64_t>(index.d_vectors.size()));

    for (const auto& [id, point] : index.d_vectors) {
        detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(id));
        detail::write_bytes(ofs, point.values.data(), index.d_dim);
    }
}

IndexFlat
load_flat_index(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error(
          "genivf::io::load_flat_index: cannot open file: " + path.string());
    }

    uint8_t magic[4]{};
    detail::read_bytes(ifs, magic, sizeof(magic));
    if (magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
        magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
        throw std::runtime_error(
          "genivf::io::load_flat_index: invalid magic — not a GIVF file");
    }

    const auto version = detail::read_val<uint8_t>(ifs);
    if (version != 2) {
        throw std::runtime_error(
          "genivf::io::load_flat_index: unsupported file version " +
          std::to_string(version) + " (expected 2 for flat index)");
    }

    const auto dim = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
    const auto num_vectors =
      static_cast<size_t>(detail::read_val<uint64_t>(ifs));

    IndexFlat index(dim, num_vectors);

    for (size_t v = 0; v < num_vectors; ++v) {
        const auto pid = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        std::vector<uint8_t> vals(dim);
        detail::read_bytes(ifs, vals.data(), dim);
        index.d_vectors.emplace(pid, Point(pid, std::move(vals)));
    }

    if (index.d_vectors.size() != num_vectors) {
        throw std::runtime_error(
          "genivf::io::load_flat_index: vector count mismatch in file");
    }

    index.d_ntotal = index.d_vectors.size();
    return index;
}

} // namespace genivf::io
