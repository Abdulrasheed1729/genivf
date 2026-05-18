#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "io.hpp"
#include "doctest.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

// Builds a small, fully trained and populated index that is reused across
// multiple test cases.
genivf::IndexIVF
make_test_index()
{
    genivf::IndexIVF ivf(2, 1, 42);

    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    ivf.train(points);
    ivf.add(points);
    return ivf;
}

// RAII wrapper that deletes a temporary file on destruction.
struct TempFile
{
    std::filesystem::path path;
    explicit TempFile(std::string name)
      : path(std::filesystem::temp_directory_path() / std::move(name))
    {
    }
    ~TempFile() { std::filesystem::remove(path); }
};

} // anonymous namespace

TEST_CASE("save_index: throws if index is not trained")
{
    genivf::IndexIVF untrained(2, 1);
    TempFile tmp("givf_test_untrained.givf");
    CHECK_THROWS_AS(genivf::io::save_index(untrained, tmp.path),
                    std::invalid_argument);
}

TEST_CASE("save_index: creates file on disk")
{
    auto ivf = make_test_index();
    TempFile tmp("givf_test_create.givf");
    genivf::io::save_index(ivf, tmp.path);
    CHECK(std::filesystem::exists(tmp.path));
    CHECK_GT(std::filesystem::file_size(tmp.path), static_cast<uintmax_t>(0));
}

TEST_CASE("save_index: file begins with GIVF magic bytes")
{
    auto ivf = make_test_index();
    TempFile tmp("givf_test_magic.givf");
    genivf::io::save_index(ivf, tmp.path);

    std::ifstream ifs(tmp.path, std::ios::binary);
    REQUIRE(ifs.is_open());
    uint8_t magic[4]{};
    ifs.read(reinterpret_cast<char*>(magic), 4);
    CHECK_EQ(magic[0], 0x47); // 'G'
    CHECK_EQ(magic[1], 0x49); // 'I'
    CHECK_EQ(magic[2], 0x56); // 'V'
    CHECK_EQ(magic[3], 0x46); // 'F'
}

TEST_CASE("save_index: throws if path directory does not exist")
{
    auto ivf = make_test_index();
    CHECK_THROWS_AS(genivf::io::save_index(ivf, "/no_such_directory/test.givf"),
                    std::runtime_error);
}

TEST_CASE("load_index: throws if file does not exist")
{
    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)genivf::io::load_index("/no_such_file.givf"),
                    std::runtime_error);
}

TEST_CASE("load_index: throws on bad magic bytes")
{
    TempFile tmp("givf_test_bad_magic.givf");

    // Write a file that starts with garbage bytes instead of 'GIVF'.
    {
        std::ofstream ofs(tmp.path, std::ios::binary);
        const uint8_t bad[4] = { 0xDE, 0xAD, 0xBE, 0xEF };
        ofs.write(reinterpret_cast<const char*>(bad), 4);
    }

    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)genivf::io::load_index(tmp.path), std::runtime_error);
}

TEST_CASE("load_index: throws on unsupported version")
{
    TempFile tmp("givf_test_bad_version.givf");

    // Correct magic, wrong version byte.
    {
        std::ofstream ofs(tmp.path, std::ios::binary);
        const uint8_t magic[4] = { 0x47, 0x49, 0x56, 0x46 };
        ofs.write(reinterpret_cast<const char*>(magic), 4);
        const uint8_t bad_ver = 99;
        ofs.write(reinterpret_cast<const char*>(&bad_ver), 1);
    }

    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)genivf::io::load_index(tmp.path), std::runtime_error);
}

TEST_CASE("load_index: throws on truncated file")
{
    TempFile tmp("givf_test_truncated.givf");

    // Write only the 4-byte magic + version — nothing else.
    {
        std::ofstream ofs(tmp.path, std::ios::binary);
        const uint8_t hdr[5] = { 0x47, 0x49, 0x56, 0x46, 0x01 };
        ofs.write(reinterpret_cast<const char*>(hdr), 5);
    }

    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)genivf::io::load_index(tmp.path), std::runtime_error);
}

TEST_CASE("round-trip: loaded index is_trained")
{
    auto original = make_test_index();
    TempFile tmp("givf_test_rt_trained.givf");

    genivf::io::save_index(original, tmp.path);
    const auto loaded = genivf::io::load_index(tmp.path);

    CHECK(loaded.is_trained());
}

TEST_CASE("round-trip: search results are identical before and after save/load")
{
    auto original = make_test_index();
    TempFile tmp("givf_test_rt_search.givf");

    genivf::Point query(0, { 0x00 });
    const auto before =
      original.search(query, 4, 2, genivf::MetricType::HAMMING);

    genivf::io::save_index(original, tmp.path);
    const auto loaded = genivf::io::load_index(tmp.path);
    const auto after = loaded.search(query, 4, 2, genivf::MetricType::HAMMING);

    REQUIRE_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK_EQ(before[i].id, after[i].id);
        CHECK_EQ(before[i].distance, doctest::Approx(after[i].distance));
    }
}

TEST_CASE("round-trip: Jaccard search results are identical")
{
    auto original = make_test_index();
    TempFile tmp("givf_test_rt_jaccard.givf");

    genivf::Point query(0, { 0xAA });
    const auto before =
      original.search(query, 4, 2, genivf::MetricType::JACCARD);

    genivf::io::save_index(original, tmp.path);
    const auto loaded = genivf::io::load_index(tmp.path);
    const auto after = loaded.search(query, 4, 2, genivf::MetricType::JACCARD);

    REQUIRE_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK_EQ(before[i].id, after[i].id);
        CHECK_EQ(before[i].distance, doctest::Approx(after[i].distance));
    }
}

TEST_CASE("round-trip: save/load is idempotent (double round-trip)")
{
    auto original = make_test_index();
    TempFile tmp1("givf_test_rt_idem1.givf");
    TempFile tmp2("givf_test_rt_idem2.givf");

    genivf::io::save_index(original, tmp1.path);
    const auto loaded1 = genivf::io::load_index(tmp1.path);

    genivf::io::save_index(loaded1, tmp2.path);
    const auto loaded2 = genivf::io::load_index(tmp2.path);

    genivf::Point query(0, { 0x00 });
    const auto r1 = loaded1.search(query, 4, 2, genivf::MetricType::HAMMING);
    const auto r2 = loaded2.search(query, 4, 2, genivf::MetricType::HAMMING);

    REQUIRE_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < r1.size(); ++i) {
        CHECK_EQ(r1[i].id, r2[i].id);
        CHECK_EQ(r1[i].distance, doctest::Approx(r2[i].distance));
    }
}

TEST_CASE("round-trip: add() works on a loaded index")
{
    // A loaded index must accept new points, just like a freshly trained one.
    auto original = make_test_index();
    TempFile tmp("givf_test_rt_add.givf");

    genivf::io::save_index(original, tmp.path);
    auto loaded = genivf::io::load_index(tmp.path);

    std::vector<genivf::Point> new_points = {
        { 10, { 0x0F } },
        { 11, { 0xF0 } },
    };
    CHECK_NOTHROW(loaded.add(new_points));
}
