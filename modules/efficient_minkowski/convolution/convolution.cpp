#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCBlas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h> 
#include <pybind11/pybind11.h>
#include <chrono>
#include <algorithm>

void ConvolutionForwardKernelGPU(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const int* neighbor_map,
    const int* neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream);

// interesting fact: Chris Choy is using double to store everything in the backward pass
// in Minkowsi Engine.
void ConvolutionBackwardKernelGPU(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, float *d_kernel,
    float *d_grad_kernel, const int * neighbor_map,
    const int * neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream);

void scatter_launch(const int n_in, const int n_out, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void gather_launch(const int n_k, const int n_in, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map, 
                           at::Tensor neighbor_offset, const bool transpose) {
    
  if (in_feat.size(1) != kernel.size(1)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  int out_nrows = out_feat.size(0);
  out_feat.resize_({out_nrows, kernel.size(2)});
  out_feat.zero_();
  
   
  int kernel_volume = kernel.size(0);
  int in_buffer_size = 1;
  bool flag = false;
  // memory optimization
  if(kernel_volume % 2 && out_nrows == in_feat.size(0)){
      flag = true;
      in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(), 
                                        neighbor_offset.data_ptr<int>() + kernel_volume/2);
      in_buffer_size = std::max(in_buffer_size, 
                           *std::max_element(neighbor_offset.data_ptr<int>() + kernel_volume/2+1, 
                           neighbor_offset.data_ptr<int>() + kernel_volume));
      in_buffer_size = std::max(in_buffer_size, 1);
      torch::mm_out(out_feat, in_feat, kernel[kernel_volume / 2]);
  }
  else{
      in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(), 
                                        neighbor_offset.data_ptr<int>() + kernel_volume);
  }
  
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  auto in_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
  auto out_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);
  int cur_offset = 0;
  for(int i = 0; i < kernel_volume; i++){
      if(flag && (i == kernel_volume / 2)){
          cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
          continue;
      }
      auto out_buffer_activated =
        torch::from_blob(out_buffer.data_ptr<float>(), 
                         {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
      auto in_buffer_activated =
        torch::from_blob(in_buffer.data_ptr<float>(), 
                         {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
      // gather
      gather_launch(in_buffer_activated.size(0), in_feat.size(0), kernel.size(1),
                   in_feat.data_ptr<float>(), in_buffer_activated.data_ptr<float>(), 
                    neighbor_map.data_ptr<int>() + cur_offset, transpose);
      // GEMM
      torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);
      // scatter
      scatter_launch(neighbor_offset.data_ptr<int>()[i], out_nrows, kernel.size(2), out_buffer_activated.data_ptr<float>(), 
                     out_feat.data_ptr<float>(), neighbor_map.data_ptr<int>() + cur_offset, transpose);
      cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
  }
  
  
  
  /*
  cublasHandle_t handle =
      //THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
      at::cuda::getCurrentCUDABlasHandle();

  ConvolutionForwardKernelGPU(
      in_feat.data_ptr<float>(), in_feat.size(1), out_feat.data_ptr<float>(),
      out_feat.size(1), kernel.data_ptr<float>(), neighbor_map.data_ptr<int>(), 
      neighbor_offset.data_ptr<int>(), in_feat.size(0), out_feat.size(0), 
      kernel.size(0), transpose, handle, 
      at::cuda::getCurrentCUDAStream());
  
  */ 

}

void ConvolutionBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const bool transpose) {
  
  grad_in_feat.resize_as_(in_feat);
  grad_in_feat.zero_();
  grad_kernel.resize_as_(kernel);
  grad_kernel.zero_();
  
  int kernel_volume = kernel.size(0);
  bool flag = false;
  int in_buffer_size;
  in_buffer_size = *std::max_element(neighbor_offset.data_ptr<int>(), 
                                        neighbor_offset.data_ptr<int>() + kernel_volume);
  
  auto options =
      torch::TensorOptions().dtype(in_feat.dtype()).device(in_feat.device());
  auto in_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
  auto in_grad_buffer = torch::zeros({in_buffer_size, in_feat.size(1)}, options);
  auto out_grad_buffer = torch::zeros({in_buffer_size, kernel.size(2)}, options);
  
  
  int cur_offset = 0;
  for(int i = 0; i < kernel_volume; i++){
      auto kernel_grad_buffer = grad_kernel[i];
      if(flag && (i == kernel_volume / 2)){
          cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
          continue;
      }
      auto out_grad_buffer_activated =
        torch::from_blob(out_grad_buffer.data_ptr<float>(), 
                         {neighbor_offset.data_ptr<int>()[i], kernel.size(2)}, options);
      auto in_grad_buffer_activated =
        torch::from_blob(in_grad_buffer.data_ptr<float>(), 
                         {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
      auto in_buffer_activated =
        torch::from_blob(in_buffer.data_ptr<float>(), 
                         {neighbor_offset.data_ptr<int>()[i], in_feat.size(1)}, options);
      // gather
      
      gather_launch(out_grad_buffer_activated.size(0), grad_out_feat.size(0), kernel.size(2),
                   grad_out_feat.data_ptr<float>(), out_grad_buffer_activated.data_ptr<float>(), 
                    neighbor_map.data_ptr<int>() + cur_offset, !transpose);
      
      gather_launch(in_buffer_activated.size(0), in_feat.size(0), kernel.size(1),
                   in_feat.data_ptr<float>(), in_buffer_activated.data_ptr<float>(), 
                    neighbor_map.data_ptr<int>() + cur_offset, transpose);
      
      // GEMM
      //torch::mm_out(out_buffer_activated, in_buffer_activated, kernel[i]);
      torch::mm_out(in_grad_buffer_activated, out_grad_buffer_activated, torch::transpose(kernel[i], 0, 1));
      torch::mm_out(kernel_grad_buffer, torch::transpose(in_buffer_activated, 0, 1), out_grad_buffer_activated);
      // scatter
      //grad_kernel[i] = kernel_grad_buffer;
      
      scatter_launch(neighbor_offset.data_ptr<int>()[i], in_feat.size(0), kernel.size(1), in_grad_buffer_activated.data_ptr<float>(), 
                     grad_in_feat.data_ptr<float>(), neighbor_map.data_ptr<int>() + cur_offset, !transpose);
      
      cur_offset += 2 * neighbor_offset.data_ptr<int>()[i];
      
  }
  
  /*  
  cublasHandle_t handle =
      //THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
      at::cuda::getCurrentCUDABlasHandle();
  ConvolutionBackwardKernelGPU(
      in_feat.data_ptr<float>(), grad_in_feat.data_ptr<float>(), in_feat.size(1),
      grad_out_feat.data_ptr<float>(), grad_out_feat.size(1), kernel.data_ptr<float>(),
      grad_kernel.data_ptr<float>(), neighbor_map.data_ptr<int>(), neighbor_offset.data_ptr<int>(), 
      in_feat.size(0), grad_out_feat.size(0), kernel.size(0), 
      transpose, handle, at::cuda::getCurrentCUDAStream());
  */
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ConvolutionForwardGPU, "point cloud convolution forward (CUDA)");
  m.def("backward", &ConvolutionBackwardGPU, "point cloud convolution backward (CUDA)");
}
