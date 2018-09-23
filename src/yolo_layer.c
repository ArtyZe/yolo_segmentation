#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    n = 1;
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
   // l.c = n*(classes + 4 + 1);
    l.c = classes;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));

//    classes=1;                              //the output size is independed by the author
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = l.h*l.w*classes; //one is the actor that if the point is optic or not
    l.inputs = l.outputs;
//    l.truths = 416*416*(2 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.loss = calloc(l.inputs*batch, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.loss_gpu = cuda_make_array(l.loss, l.inputs*batch);
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo  l.outputs= %d l.c is %d\n", l.outputs, l.c);
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w;
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

float get_yolo_mask(float *x, int index, int i, int j)
{
    float mask_pred_ij = x[index];
    //b.y = (j + x[index + 1*stride]) / lh;
    //b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    //b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return mask_pred_ij;
}

void delta_yolo_seg(float truth, float pred, int index, int i, int j, int w, int h, float *delta, int stride)
{
//    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
//    float iou = box_iou(pred, truth);

//    float tx = (truth.x*lw - i);
//   float ty = (truth.y*lh - j);
//   float tw = log(truth.w*w / biases[2*n]);
//    float th = log(truth.h*h / biases[2*n + 1]);
      delta[index] = abs(truth - pred);
//    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
//    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
//    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
//    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
//    return iou;
}

      //delta_yolo_class(l.output, l.delta, class_index, class_truth, l.classes, l.w*l.h, 0);

void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *loss)
{
    int n;
    //printf("the current truth class is %d\n",classes);
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] =-((n == class)?1 : 0) + output[index + stride*n];
        loss[index + stride*n] = (((n == class)?1 : 0)-class)*log(1-output[index + stride*n]+.0000001);

//       delta[index + stride*n] = ((n != class)?1 : 0) - ((log(output[index + stride*n])>100)?100:log(output[index + stride*n]) );
//    printf("the current output is %f\n",log(output[index + stride*n]));
    }
     //if(class == 1){
     //delta[index] = delta[index]*1.1;
////     printf("delta is %f\n",delta[index]);
     //}
}

static int entry_index(layer l, int batch, int location, int entry)
{
//    int n =   location / (l.w*l.h);                   //n=1
//        int loc = location % (l.w*l.h);                 //loc =j*l.w+i
 //   return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
        return batch*l.outputs+location*l.classes+entry;
//   return batch*l.outputs+loc+entry*l.w*l.h;

}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b;

    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    int k;
    for(k = 0; k < l.w*l.h*2; ++k){
        int index = k;
        l.delta[index] = 0 - l.output[index]; //对float l.delta[0]至float l.delta[l.w*l.h*l.classes-1]先验设置（l.delta[]和l.output[]有相同的结构）

    }

#if 1
    for (b = 0; b < l.batch; ++b){
            //int index = entry_index(l, b, l.w*l.h, 0);
            int index = b*l.w*l.h;
            activate_array(l.output + index, l.w*l.h*2, LOGISTIC);
//            index = entry_index(l, b, l.w*l.h, 1);
//            activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
    }
#endif


    if(!net.train) return;
    int count = 0;
    *(l.cost) = 0;
    float largest = -FLT_MAX;

    for(i = 0; i < l.w*l.h; ++i){
        if(net.input[i] > largest) net.input[i] = largest;
    }
    int count_pixel = 0;
    for (b = 0; b < l.batch; b++) {
        for (j = 0; j < l.h*l.w; j++) {
              int class_truth_index = b*l.w*l.h*3+j;
              float class_truth = net.truth[class_truth_index]; //0,1,2,.....19
              int class_index = b*l.w*l.h+j;
#if 0
                  sum = exp(l.output[class_index]-largest)+exp(l.output[class_index+l.w*l.h]-largest);
                  l.output[class_index] = exp(l.output[class_index]-largest)/sum;
                  if(l.output[class_index]<FLT_MIN) l.output[class_index] = FLT_MIN;
                  l.output[class_index+l.w*l.h] = exp(l.output[class_index+l.w*l.h]-largest)/sum;
                  if(l.output[class_index+l.w*l.h]<FLT_MIN) l.output[class_index+l.w*l.h] = FLT_MIN;
#endif
//        printf("l.w is %d, l.h is %d\n",l.w,l.h);
 //       printf("the output x of the point is %d, y of the point is %d\n",i,j);
#if 0
      //if(class_truth ==1){
        //printf("the truth x of the point is %f, y of the point is %f\n",net.truth[class_truth_index+2],net.truth[class_truth_index+1]);
        //printf("the output x of the point is %f, y of the point is %f\n",i,j);
        printf("the truth class_truth of the point is %f\n",class_truth);
        printf("the  class_output of the point is %f\n",l.output[class_index]);

       // count_pixel = ++;
      //  delta_yolo_class(l.output, l.delta, class_index, class_truth, l.classes, l.w*l.h, 0);

       //      printf("the  class_output of the point is %f\n",l.output[class_index+l.w*l.h]);
       //}
#endif
        delta_yolo_class(l.output, l.delta, class_index, class_truth, l.classes, l.w*l.h, l.loss);
            //if(class_truth){        // only calculate delta for positive pixel
              //l.delta[class_index] = class_truth - l.output[class_index];// renew the value of l.delta[index]

                //}
              }
        }
    *(l.cost) = pow(mag_array(l.loss, l.outputs * l.batch), 2);
    //sum_array(l.delta, l.outputs*l.batch)/2;
    //
   //    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
    printf("The number of positive pixel is %d\n", count_pixel);
    printf("YOLO ----- l.cost  %f,  Avg Cost: %f\n", count, *(l.cost), *(l.cost)/(512*1024));
}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,num;
    float *predictions = l.output;
//    if (l.batch == 2) avg_flipped_yolo(l);    dont know how it works, but you can try to open it first
    int count = 0;
    for (j = 0; j < l.h; j++) {
        for (i = 0; i < l.w; i++) {
        //int obj_index  = entry_index(l, 0, i, 4);
        //float objectness = predictions[obj_index];
        //if(objectness <= thresh) continue;
//        dets[count].seg = get_yolo_mask(predictions, class_index, l.w, l.h);
        float max_val = 0.5;
        int class_index = entry_index(l, 0, j*l.w+i, 0);
        for(num=0; num<2; num++){

        if(get_yolo_mask(predictions, class_index, l.w, l.h) > max_val){
          dets[count].classes = num;
          max_val = get_yolo_mask(predictions, class_index, l.w, l.h);
        }
        else{
          dets[count].classes = 0;
        }
    }
        if(get_yolo_mask(predictions, class_index, l.w, l.h) > max_val){
        printf("count is %d,class is %d\n",count, dets[count].classes);
      }
      count++;
      }
    }
    //correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;    //count = l.w * l.h
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
#if 1
    for (b = 0; b < l.batch; ++b){
        int index = b*l.w*l.h;
        activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
    }
#endif
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
    //float smallst = FLT_MIN;
    //for(i = 0; i < l.w*l.h*2; i++){
        //if(net.input[i] > largest) largest = net.input[i];
    //}
    //yolo_kernel<<<cuda_gridsize(batch*l.w*l.h*l.c), BLOCK>>>(net.input_gpu, l.w, l.h, l.c, l.output_gpu, net.truth_gpu, l.loss_gpu, l.delta_gpu);
    //cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
    //l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    //int b, j, i, sum, n;
    //float largest = -FLT_MAX;
    //for(i = 0; i < l.w*l.h*2; i++){
        //if(net.input[i] > largest) largest = net.input[i];
    //}
    //cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //for (b = 0; b < l.batch; b++) {
        //for (j = 0; j < l.h; j++) {
            //for (i = 0; i < l.w; i++) {
                //int class_truth_index = b*l.w*l.h+j*l.w+i;
                //int class_truth = net.truth[class_truth_index]; //0,1,2,.....19
                //int class_index = b*l.w*l.h*2+j*l.w+i;
                //sum = exp(l.output[class_index]-largest)+exp(l.output[class_index+l.w*l.h]-largest);
                //l.output[class_index] = exp(l.output[class_index]-largest)/sum;
                //l.output[class_index+l.w*l.h] = exp(l.output[class_index+l.w*l.h]-largest)/sum;
                //for(n = 0; n < l.classes+1; n++){
                    //l.delta[class_index + l.w*l.h*n] =l.output[class_index + l.w*l.h*n]-1;
                //}
            //}
        //}
    //}
    //cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

