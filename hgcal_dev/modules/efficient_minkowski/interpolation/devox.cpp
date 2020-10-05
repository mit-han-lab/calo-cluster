#include <torch/torch.h>

#include <vector>


//CUDA forward declarations
void devoxelize_wrapper(int N, int c, const int * indices, const float * weight, const float * feat, float * out);
void devoxelize_grad_wrapper(int N, int n, int c, const int * indices, const float * weight, const float * top_grad, float * bottom_grad);


//make sure indices is int type
//feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor devoxelize_forward(
    const at::Tensor feat,
    const at::Tensor indices,
    const at::Tensor weight
)
{  
  int b = feat.size(0);
  //printf("%d\n", b);
  int c = feat.size(1);
  int N = indices.size(0);
  
  at::Tensor out = torch::zeros({N, c}, at::device(feat.device()).dtype(at::ScalarType::Float));
  devoxelize_wrapper(N, c, indices.data_ptr<int>(), weight.data_ptr<float>(), feat.data_ptr<float>(), out.data_ptr<float>());
  return out;
}
    

//top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad: (b,c,s), s=r^3
at::Tensor devoxelize_backward(
    const at::Tensor top_grad,
    const at::Tensor indices,
    const at::Tensor weight,
    int n
)
{
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad = torch::zeros({n, c}, at::device(top_grad.device()).dtype(at::ScalarType::Float));
  devoxelize_grad_wrapper(N, n, c, indices.data_ptr<int>(), weight.data_ptr<float>(), top_grad.data_ptr<float>(), bottom_grad.data_ptr<float>());
  return bottom_grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &devoxelize_forward, "Devoxelization forward (CUDA)");
  m.def("backward", &devoxelize_backward, "Devoxelization backward (CUDA)");
}




