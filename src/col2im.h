#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(float* data_col,
        int channels, int height, int width,
<<<<<<< HEAD
        int ksize, int stride, int pad, int dilation, float* data_im);
=======
        int ksize, int stride, int pad, float* data_im);
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee

#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
<<<<<<< HEAD
        int ksize, int stride, int pad, int dilation, float *data_im);
=======
        int ksize, int stride, int pad, float *data_im);
>>>>>>> d2bad383be6fc51a225bdc438fe8661eec5816ee
#endif
#endif
