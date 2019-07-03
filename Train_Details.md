 The file is to introduce how to make change in autor's original code to make semantice segmentation with Yolo. But I have some points to verify:   
    
    1. train multiply classes ;  
    2. train instance segmentation.
    
 I will update the file and upload code step by step :)
     
    1.modification of Main program   
    2.modification of Origin image and Lable image read mode   
    3.modification of Loss function  
    4.modification of Network structure   
    5.Revision experience and error correction summary

###1.modification of Main program

Mainly according to my training category, set the classes to 1. In fact, in the process of code modification, this parameter has no effect. 
In the instance segmentation process, it will affect how many layers of l.w × l.h are generated. But in order to let everyone know what role this parameter will play, 
here is the corresponding modification. As for the threads, it is mainly related to your ngpus. In the code behind, I found that my image and label are loaded once in each thread. 
However, if I press the author's 32 threads, it means that I have 31 threads which do nothing. Through the printf imgs this parameter also verified my thoughts.

    void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
    {
    args.classes = 1;
    args.threads = 1;
    }

    void *load_threads(void *ptr)
    {
        for(i = 0; i < args.threads; ++i){
            args.d = buffers + i;
            args.n = (i+1) * total/args.threads - i * total/args.threads;
            threads[i] = load_data_in_thread(args);
        }
    }


###2.modification of Origin image and Lable image read mode

In the original author's code, it maybe come from the idea of the coco dataset, and needs to cooperate with the json file. 
I use the main code as an example.

    image orig = load_image_color(random_paths[i], 0, 0);
    image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);

It can be seen that the image is mainly read by two functions and returned to the image structure object.

    image get_segmentation_image(char *path, int w, int h, int classes)
    {    
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        image mask = make_image(w, h, classes);
        FILE *file = fopen(labelpath, "r");
        if(!file) file_error(labelpath);
        image part = make_image(w, h, 1);
        while(fscanf(file, "%d %s", &id, buff) == 2){
            int n = 0;
            int *rle = read_intlist(buff, &n, 0);
            load_rle(part, rle, n);
            or_image(part, mask, id);
            free(rle);
        }
        return mask;
    }

Do a few cuts, please ignore the syntax error, you can see that each input image, you need a corresponding .txt file, the specific format of this file and so on.


You can see that the last mask object returned  is an image of l.w×l.h×classes, in author's original code, the classes needs to be modified according to its own training set,
 or modified in the txt file. Let's take a look at the subfunctions below:  
 
   
     void load_rle(image im, int *rle, int n)
    {
        int count = 0;
        int curr = 0;
        int i,j;
        for(i = 0; i < n; ++i){
            for(j = 0; j < rle[i]; ++j){
                im.data[count++] = curr;
            }
            curr = 1 - curr;
        }
        for(; count < im.h*im.w*im.c; ++count){
            im.data[count] = curr;
        }
    }

 The work done here is to assign a value of 0-1 to each part image according to the first and last parameters of each line in the txt file.
 
 In fact, it is very simple, txt is the first and last coordinates of each line stored, so you can set the value of 0-x2 to 1,
 and then set the value of 0-x1 to 0, so that the vaule of x1-x2 will be assign to 1 successful, the other lines are same;
 However, it should be noted that the whole image is actually a one-dimensional vector of l.w*l.h, which is just for everyone to explain.
 
     void or_image(image src, image dest, int c)
    {
        int i;
        for(i = 0; i < src.w*src.h; ++i){
            if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
        }
    }
 
 Here is simple, according to the part image of each category, the corresponding layer in the corresponding mask image is set to 0 or 1. 
 After understanding the author's code, I feel that I need to enter the label image to  generate the label matrix code automatically, so I will proceed to modify it.
 
     image mask = load_image_gray(random_paths[i], orig.w, orig.h);
     First change the input mode of the mask as above
     image load_image_gray(char *path, int w, int h){
        char labelpath[4096];
        find_replace(labelpath, "_leftImg8bit.png", "_gtFine_instanceIds.png", labelpath);
        find_replace(path, "images", "mask", labelpath);
        find_replace(labelpath, "JPEGImages", "mask", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPG", ".txt", labelpath);
        find_replace(labelpath, "_leftImg8bit.png", "_gtFine_instanceIds.png", labelpath);
        return load_image(labelpath, w, h, 1);
    }
 
 This is very simple, I believe everyone can understand, is based on my _leftImg8bit.png find the corresponding label image _gtFine_instanceIds.png,
 and then import through load_image, if you use my framework, you must modify parameters according to your own data set on the find_replace function.
 Otherwise, the corresponding image cannot be found, resulting d.x and d.y in the data structure being 0. Second, the error that can not find file is reported.
 
 
 There is a very important change to be made here: the author's normalization of the image

In load_image, it was found that the author did not read the original pixel value of each channel, but divided by 255. But through the study of the previous code, 
the value in the dy matrix is 0-1, so if I follow the author's way to read the image, my value is 0-1/255, so I need to modify the code here. 
Determine if the label image is being read (because my label image is grayscale, channel == 0)
 
     image im = make_image(w, h, c);
        if(c==1){
          for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
              for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index];
              }
            }
          }
        }else{
          for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
              for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
              }
            }
          }
        }
 

The second important change: local bilinear interpolation of the original image and the label image

I believe that students who have read the code should know that the author does not directly use the training image after reading the input image. 
In addition to normalization, the image is cropped, bilinear interpolation and random rotation, which may have no effect on the original image. 
However, the effect on the 0-1 mask image is large, which causes some 1 internal dot interpolation to be called 0, or some edge values are not 0-1 but intermediate values.
 
     image rotate_crop_image_seg(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
    {
        int x, y, c;
        float cx = im.w/2.;
        float cy = im.h/2.;
        image rot = make_image(w, h, im.c);
        for(c = 0; c < im.c; ++c){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                    float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                    float val = bilinear_interpolate(im, rx, ry, c);
                    //if(val!=0) printf("the value is %f\n",val);
                    //if(val >=0.25)   //给定一个阈值，判断插值后的每个点像素值
                    //{
                    //  val = 1;
                    //}else{
                    //  val = 0;
                    //}
                    set_pixel(rot, x, y, c, val);
                }
            }
        }
        return rot;
    }
 
 The part I commented out in the above code is the judgment statement, and the image that guarantees bilinear interpolation is also only 0-1 image. 
 However, I found out in the experiment that in fact, according to the author, the improvement is not carried out, and the effect can also occur, but there will be some noise.

###3.modification of Loss function

In fact, this is not a modification, it should be a choice. The author has given a variety of options, here to give you a list: 
1 logistic; 
2 softmax layer, compared to logistic, mainly for more than one category of classification, I believe that students who understand the softmax function should understand;
3 cost layer, here can be set to seg to ensure that it is used for semantic segmentation, of course, the author here naughty said that he himself feels that the loss function here seems to be a problem.

Here is a more important point. I believe that it is unknow for many students: in the calculation of all the semantics of darknet, 
the author’s delta is the same as the value of the backstep gradient used for weights update. It is truth-pred, but different loss calculation methods are different. 
So what is the difference between these two values, loss is to show you how much the model fits, and delta is really used for weight update.

    void backward_cost_layer_gpu(const cost_layer l, network net)
    {
        axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1); //将l.delta_gpu拷贝给net.delta_gpu
    }

Students who don't believe can go to the update function of the convolutional layer and the network.c file. 
By the way, understand the gradient backcast and training methods of darknet :)


###4. cfg --- the network structure
There is nothing special to say here, you can write according to various network structures such as UNET or SegNet.


###5. Revision experience and error correction summary
Let us talk about the more common mistakes:

 #####CUDA OUT OF MEMORY

 The reason is very clear, that is, your gpu is weak, the memory is too small, what you need to do at this time is to modify your network, delete the number of nodes or layers; 
 then set the batch to 1, subdivision to 1

 
 #####Segmentation fault:

 The first possibility is to access the out-of-bounds. 
 For example, if your image is l.w×l.h, the pointer you access to l.w×l.h+1 will cause this error; 
 the second possibility is that you directly access the GPU. Data, this is also a mistake often made by students who don't understand the darknet framework. 
 The code example that is modified and returned after correct access to the GPU data is given below.
 
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    image im = make_image(1024, 512, 1);
    for(i=0; i<l.w*l.h; i++){
        l.delta[i] = 0 - net.input[i];
        im.data[i] = (float)net.input[i];
    }
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    save_image(im, "feature_map");
    free_image(im);

Here, the output matrix of the layer is saved as an image. What needs to be done is to first pull the data in the l.output_gpu, that is, the GPU, pull to the CPU, 
and then perform the operation, and then if necessary, return the delta in the calculated CPU to the GPU for back-transmission, you need to use the PUSH function to push it back,
 but you can't directly save l.output_gpu as an image,  it will report the segmentation fault error if you do that.

Let me talk about some training points:

#####Learning rate

Mainly depends on the proportion of the positive sample pixel of an image. If it is high, it can be larger. If it is smaller, it should be smaller. I set it to 10-7.

#####Training loss does not fall

In fact, during my training, the loss value has been hopping, but at the beginning there is a downward trend; 
what needs to be done here is to save the output image of your last layer and compare it with your input image to see the heat map. Is it similar to yours? 
If it is similar, it doesn't matter and keep training. Look at the results at the end, you must patiently train thousands of tens of thousands of steps to see the results.
Don't stop train and modify the code when you see the loss value doesn't descend at the first. You can train safely if in the premise that I said.

