#include <torch/torch.h>

#include <vector>


//CUDA forward declarations
void insertion_wrapper(int N, int c, int s, const float * data, const int * idx, const int * counts, float * out);
void insertion_grad_wrapper(int N, int c, int s, const float * top_grad, const int * idx, const int * counts, float * bottom_grad);
//make sure indices is int type
//feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n) (preprocessed indices)

at::Tensor insertion_forward(
    const at::Tensor inputs,
    const at::Tensor idx,
    const at::Tensor counts
)
{
  //return group_point_forward_gpu(points, indices);
  
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);
  at::Tensor out = torch::zeros({N1, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  insertion_wrapper(N, c, N1, inputs.data_ptr<float>(), idx.data_ptr<int>(), counts.data_ptr<int>(), out.data_ptr<float>());
  return out;
}


at::Tensor insertion_backward(
    const at::Tensor top_grad,
    const at::Tensor idx,
    const at::Tensor counts,
    const int N
)
{
  //return group_point_forward_gpu(points, indices);
  
  int c = top_grad.size(1);
  int N1 = counts.size(0);
  at::Tensor bottom_grad = torch::zeros({N, c}, at::device(idx.device()).dtype(at::ScalarType::Float));
  insertion_grad_wrapper(N, c, N1, top_grad.data_ptr<float>(), idx.data_ptr<int>(), counts.data_ptr<int>(), bottom_grad.data_ptr<float>());
  return bottom_grad;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("insertion_forward", &insertion_forward, "Insertion forward (CUDA)");
  m.def("insertion_backward", &insertion_backward, "Insertion backward (CUDA)");
}


