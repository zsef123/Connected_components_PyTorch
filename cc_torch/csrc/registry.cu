#include <torch/script.h>
#include <torch/extension.h>

#include "buf.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cc_2d", &connected_componnets_labeling_2d, "connected_componnets_labeling_2d");
  m.def("cc_3d", &connected_componnets_labeling_3d, "connected_componnets_labeling_d");
}