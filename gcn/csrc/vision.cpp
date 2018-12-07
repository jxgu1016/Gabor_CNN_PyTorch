#include "GOF.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gof_forward", &GOF_forward, "GOF forward");
  m.def("gof_backward", &GOF_backward, "GOF backward");
}
