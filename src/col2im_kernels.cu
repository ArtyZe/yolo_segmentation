#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "col2im.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
<<<<<<< HEAD
        const int height_col, const int width_col, const int dilation,
=======
        const int height_col, const int width_col,
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee
        float *data_im) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
<<<<<<< HEAD
        int w_col_start = (w < ksize) ? 0 : (w - ((ksize-1)*dilation+1)) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ((ksize-1)*dilation+1)) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        //int offset =
            //(c * ksize * ksize + h * ksize + w) * height_col * width_col;
        //int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        //int coeff_w_col = (1 - stride * height_col * width_col);
        //for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            //for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                //val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            //}
        //}
        //data_im[index] += val;
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
		  for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
			int h_k = (h - h_col * stride);
			int w_k = (w - w_col * stride);
			if (h_k % dilation == 0 && w_k % dilation == 0) {
			  h_k /= dilation;
			  w_k /= dilation;
			  int data_col_index = (((c * ksize + h_k) * ksize + w_k) *
									height_col + h_col) * width_col + w_col;
			  val += data_col[data_col_index];
			}
		  }
		}
		data_im[index] = val;
=======
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee
    }
}

void col2im_gpu(float *data_col,
        int channels, int height, int width,
<<<<<<< HEAD
        int ksize, int stride, int pad, int dilation, float *data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;
    int width_col = (width + 2 * pad - (dilation * (ksize - 1) + 1)) / stride + 1;
=======
        int ksize, int stride, int pad, float *data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee
    int num_kernels = channels * height * width;
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, data_col, height, width, ksize, pad,
                stride, height_col,
<<<<<<< HEAD
                width_col, dilation, data_im);
=======
                width_col, data_im);
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee
}

