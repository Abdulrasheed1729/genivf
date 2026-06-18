= Report on new BSIVF nprobe based implementation

A new implementation for the bsivf index has been added. In this implementation, a new parameter `nprobe` has been added to the `bsivf` index so as to make binary search based on the number of probes.

#figure(
  image("docs/nprobe-10.pdf"),
)

#figure(
  image("docs/nprobe-15.pdf"),
)
#figure(
  image("docs/nprobe-25.pdf"),
)

In addition, to that I made calculations for recall based on a tolerance value, i.e. how many points fall within a certain distance -- tolerance -- from the ground truth. Here are the plots based on the tolerance value: $10$.

#figure(
  image("docs/bsivf_summary_nprobe_10_tol_10.pdf"),
)
#figure(
  image("docs/bsivf_summary_nprobe_15_tol_10.pdf"),
)
#figure(
  image("docs/bsivf_summary_nprobe_25_tol_10.pdf"),
)

From, the above it is quite clear that $n_("probe") = 25$ is the optimal value for the number of probes. In that case, the following plot shows recall plot for different strides but with the strides as the tolerance value.

#figure(
  image("docs/bsivf_summary_nprobe_tol_by_stride.pdf"),
)
