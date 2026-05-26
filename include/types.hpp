#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace genivf {

struct WindowMetaData
{
    std::string sequence_name;
    int start_pos = 0;
};
// Metric types for distance calculations.
enum class MetricType
{
    L2,
    HAMMING,
    JACCARD
};

// A container representing a single packed-binary datapoint.
//
// `id`: Unique identifier for the point within the index.
// `values`: The packed binary vector data.
struct Point
{
    size_t id;
    std::vector<uint8_t> values;

    Point(size_t id, std::vector<uint8_t> vals)
      : id(id)
      , values(std::move(vals))
    {
    }

    Point(size_t id, std::initializer_list<uint8_t> args)
      : id(id)
      , values(args)
    {
    }

    Point(const Point&) = default;
    Point(Point&&) = default;
    Point& operator=(const Point&) = default;
    Point& operator=(Point&&) = default;

    // Returns the packed byte at position `index` (not a bit index).
    [[nodiscard]] uint8_t operator[](size_t index) const noexcept
    {
        return values[index];
    }

    // Checked access. Throws `std::out_of_range` if `index` is out of range.
    [[nodiscard]] uint8_t at(size_t index) const
    {
        if (index >= values.size()) {
            throw std::out_of_range(
              "Point::at: index " + std::to_string(index) +
              " out of range [0, " + std::to_string(values.size()) + ")");
        }
        return values[index];
    }
};

// Simple struct representing a search result.
//
// `id`: The point id.
// `distance`: The distance to the query point.
struct SearchResult
{
    size_t id;

    double distance;

    // Natural ordering by distance (ascending).
    bool operator<(const SearchResult& other) const noexcept
    {
        return distance < other.distance;
    }
};

// A single Voronoi cell in the IVF index.
// `centroid`      — the representative binary vector for this cell.
// `point_indices` — ids of all points assigned to this cell's inverted list.
// `id`            — cell index in [0, num_cells).
struct Cluster
{
    Point centroid;
    std::vector<size_t> point_indices;
    std::vector<uint8_t> flat_vectors; // Contiguous block of vector data for
                                       // points assigned to this cell
    size_t id;

    Cluster(size_t cell_id, Point ctr)
      : centroid(std::move(ctr))
      , id(cell_id)
    {
    }
};

}
