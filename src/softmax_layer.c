#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups, int w, int h, int c)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.w = w;
    l.h = h;
    l.c = c;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    softmax_instance_folder_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.c, l.h, l.w, l.temperature, l.output_gpu);
    // softmax_instance_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.c, l.h, l.w, l.temperature, l.output_gpu);
    // resize the truth from l.w*l.h*1 to l.w*l.h*l.c
    float *truth_temp = calloc(l.c*l.w*l.h, sizeof(float));
    float *truth_temp_gpu = cuda_make_array(truth_temp, l.c*l.w*l.h); 
    int i, j;
    if(net.truth && !l.noloss){
        cuda_pull_array(net.truth_gpu, net.truth, l.batch*l.w*l.h);
        // init truth_temp to 0
        // for (i = 0; i < l.c; ++i){
        //     for (j = 0; j < l.h; ++j){
        //         for (int k = 0; k < l.w; ++k){
        //             int index = i*l.h*l.w+j*l.w+k;
        //             truth_temp[index] = 0;
        //         }
        //     }
        // }
        // set ont-hot map 
        for (i = 0; i < l.h; ++i){
            for (j = 0; j < l.w; ++j){
                int class_id = net.truth[i*l.w+j];
                // printf("truth value is %d\n", class_id);
                int value_index = class_id*l.h*l.w+i*l.w+j;
                truth_temp[value_index] = 1;
            }
        }
        
        cuda_push_array(truth_temp_gpu, truth_temp, l.batch*l.w*l.h*l.c);
    }
#if 0
    if(!net.train){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
        image im = make_image(1024, 512, 1);
        // image im_truth = make_image(1024, 512, 1);
        for(int ii=0; ii<l.w*l.h; ii++){
            int max_id = 0;
            float max_value = 0;
            for(int jj = 0; jj < l.c; ++jj){
                if(max_value < l.output[jj*l.w*l.h + ii]){
                    max_value = l.output[jj*l.w*l.h + ii];
                    max_id = jj;
                }
            }
            // printf("the id is %d\n", max_id);
            // printf("the id is %f\n", (float)max_id);
            if(max_value > 0.05){
                im.data[ii] = (float)max_id;
            }else{
                im.data[ii] = 0;
            }
            
            // im_truth.data[ii] = (float)net.truth[ii];
        }
        save_image(im, "output");
        // save_image(mask_to_rgb(im_truth), "truth");
        free_image(im);
        // free_image(im_truth);
    }
#endif
    // calculate delta and loss
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, truth_temp_gpu, l.delta_gpu, l.loss_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
    free(truth_temp);
    cuda_free(truth_temp_gpu);
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
