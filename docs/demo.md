---
title: "Result Presentation"
author: Abdulrasheed Fawole
theme:
    name: catppuccin-mocha
---

# Introduction
<!-- column_layout: [1, 2] -->
<!-- font_size: 7 -->
<!-- incremental_lists: true -->
<!-- list_item_newlines: 2 -->
<!-- column: 0 -->
- Named Genivf (Genomics + IVF)
- Similar API definition with FAISS
<!-- column: 1 -->
<!-- alignment: center -->

![image:width:100%](dna.jpg)

_Source: Andrey Prokhorov/E+/Getty Images_


<!-- end_slide -->

# Methods

Training steps
===

- Training with KMeans with L2 distance
<!-- pause -->

```md
 _______________________________        __________________________________________
|                               |      |                                          |
|  Data Points (binary vector)  |----> |     expand into floats vector of (+1/-1) |
|_______________________________|      |__________________________________________|
                                                            |
                                                            |
                                        __________________________________________
                                       |                                          |
                                       |   train with kmeans using l2 distance    |
                                       |__________________________________________|
                                                            |
                                                            |
                                         __________________________________________
                                        |                                          |
                                        | quantise centroids into binary vector    |
                                        |__________________________________________|
```

<!-- end_slide -->


# Methods
Quantisation functions (`binary_to_real`)
===

- Expands a packed binary vector into floats (+1.0 / -1.0).
<!-- pause -->
```cpp
[[nodiscard]] inline bool
binary_to_real(std::size_t d, const uint8_t* x_in, float* x_out)
{
    if (d % 8 != 0)
        return false;
    for (std::size_t i = 0; i < d; ++i) {
        x_out[i] =
          2.0f * static_cast<float>((x_in[i >> 3] >> (i & 7)) & 1u) - 1.0f;
    }
    return true;
}


```
_Source: faiss library_

<!-- end_slide -->
# Methods

Quantisation functions (`real_to_binary`)
===
- Convert a d-dimensional float vector to a packed binary vector.
<!-- pause -->
```cpp
[[nodiscard]] inline bool
real_to_binary(std::size_t d, const float* x_in, uint8_t* x_out)
{
    if (d % 8 != 0)
        return false;
    for (std::size_t i = 0; i < d / 8; ++i) {
        uint8_t byte = 0;
        for (int j = 0; j < 8; ++j) {
            if (x_in[i * 8 + j] > 0.0f) {
                byte |= static_cast<uint8_t>(1u << j);
            }
        }
        x_out[i] = byte;
    }
    return true;
}


```
_Source: faiss library_

<!-- end_slide -->

# Accuracy Measure
Recall measure for `n_list = 128`
===
<!-- column_layout: [1, 1] -->
<!-- alignment: center -->

<!-- column: 0 -->
<!-- font_size: 7 -->

Seq: `q1`

| _nprobe_ | recall|
| --- | ---|
| 1  | 35.66% |
| 2  | 45.252% |
| 4  | 52.48% |
| 8  | 59.364% |
| 16 | 65.3% |
| 32 | 70.252% |

<!-- column: 1 -->
Seq: `q2`

| _nprobe_ | recall|
| --- | ---|
| 1  | 35.66% |
| 2  | 45.252% |
| 4  | 52.48% |
| 8  | 59.364% |
| 16 | 65.3% |
| 32 | 70.252% |

<!-- reset_layout -->
<!-- end_slide -->


# Accuracy Measure
Recall measure for `n_list = 256`
===
<!-- column_layout: [1, 1] -->
<!-- alignment: center -->
<!-- column: 0 -->

Seq: `q1`

| nprobe | recall |
| ---    | ---    |
| 1      |55.692% |
| 2      |70.052% |
| 4      |76.936% |
| 8      |81.676% |
| 16     |85.748% |
| 32     |88.136% |
| 64     |89.172% |

<!-- column: 1 -->

Seq: `q2`

| nprobe | recall |
| ---    | ---    |
| 1      |56.752% |
| 2      |70.948% |
| 4      |77.472% |
| 8      |81.868% |
| 16     |85.772% |
| 32     |88.004% |
| 64     |89.104% |

<!-- reset_layout -->
<!-- end_slide -->

# Methods

Search
===

- Hamming distance is used for querying and its computed using 64-bit words


```cpp
[[nodiscard]] inline uint32_t
distance_hamming(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t d = 0;
    std::size_t i = 0;

    const std::size_t num_words = N / 8;
    if (num_words > 0) {
        const uint64_t* a_64 = reinterpret_cast<const uint64_t*>(a);
        const uint64_t* b_64 = reinterpret_cast<const uint64_t*>(b);
        for (std::size_t w = 0; w < num_words; ++w) {
            d += std::popcount(a_64[w] ^ b_64[w]);
        }
        i = num_words * 8;
    }

    for (; i < N; ++i) {
        d += std::popcount(static_cast<uint8_t>(a[i] ^ b[i]));
    }
    return d;
}

```


<!-- end_slide -->

# Accuracy Measure
Recall measure for `n_list = 512`
===
<!-- column_layout: [1, 1] -->
<!-- alignment: center -->

<!-- column: 0 -->
<!-- font_size: 7 -->

Seq: `q1`

| _nprobe_ | recall|
| --- | ---|
| 1   | 58.984% |
| 2   | 70.66% |
| 4   | 77.136% |
| 8   | 80.888% |
| 16  | 83.276% |
| 32  | 84.596% |
| 64  | 85.28% |
<!-- | 128 | 85.492% |
| 256 | 85.532% |
 -->
<!-- column: 1 -->
Seq: `q2`

| _nprobe_ | recall|
| --- | ---|
| 1   | 59.268% |
| 2   | 70.956% |
| 4   | 77.224% |
| 8   | 80.772% |
| 16  | 83.108% |
| 32  | 84.548% |
| 64  | 85.152% |
<!-- | 128 | 85.38% |
| 256 | 85.42% |
 -->
<!-- end_slide -->

# Conclusion

Next Steps
===

<!-- list_item_newlines: 4 -->
<!-- incremental_lists: true -->

- Finish up `introduction`, `background` and literature review rewrite `*`
- Binary search implementation `*`
- Other sections of essay writing `*`
- Clustering algorithm improvement (KMeans++, KMedoid (since it works with any distance function)?) `?`
- Parallelisation, SIMD and GPU `?`
