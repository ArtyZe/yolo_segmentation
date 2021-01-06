# yolo_segmentation
![GitHub stars](https://img.shields.io/github/stars/ArtyZe/yolo_segmentation) ![GitHub forks](https://img.shields.io/github/forks/ArtyZe/yolo_segmentation)  ![GitHub watchers](https://img.shields.io/github/watchers/ArtyZe/yolo_segmentation)


![](https://img.shields.io/badge/LinuxCPU-Pass-brightgreen.svg?style=plastic) ![](https://img.shields.io/badge/LinuxGPU-Pass-brightgreen.svg?style=plastic) ![](https://img.shields.io/badge/WindowsCPU-Pass-brightgreen.svg?style=plastic)

The code is to get segmentation image by darknet

In the process of my project, I have referenced nithi89/unet_darknet in some points and nithilan has give me many important  

advices, thanks to him, and if you have interest you can visit his homepage.

This is my third version, I added dilation convolutional, and now it has not so perfect result, but i think it's good enough for me. 

I will continue to update afterwards, please stay tuned.

[The Commond to Run My Project]
=========
Compile: 

	make -j8

Train: 

	./darknet segmenter train cfg/maskyolo.data cfg/instance_segment.cfg [pretrain weights file I gave to you] 

Test:

	./darknet segmenter test cfg/maskyolo.data cfg/instance_segment.cfg [weights file] [image path]

Merge two images:

	python Merge.py
	
And you will get the mask image named final.png

Test image:
Merge them together image:
<center class="half">
    <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/orig.png">
    <!-- <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/orig1.png"> -->
</center>

Output image:(for orig)<center class="half">
    <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/output.png">
    <!-- <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/output1.png"> -->
</center>

Merge them together image:(not so good, 1. more epochs; 2. deeper or more complex backbone)
<center class="half">
    <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/final.png">
    <!-- <img src="https://github.com/ArtyZe/yolo_segmentation/blob/master/result/final1.png"> -->
</center>

[Pretrain weights file and cfg file]  
========  
1. https://www.dropbox.com/sh/9wrevnyzwfv8hg7/AAA1MJElri9aROsjaPTxO5KCa?dl=0
2. https://pan.baidu.com/s/15gcrXGzb-fY2vGdl4KlLqg
   password: bk01
   
[How to Train with Your Own Dataset ?]  
========  

The Way is so easy, you only need three files:  
 
	original colorful image;  
	
	label image(pixel value is 0, 1, 2, 3 if you have 3 classes + background);  
	
	train.list.

For example with cityscape dataset:

Colorful Original Image:  
------------
![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/data/aachen_000001_000019_leftImg8bit.png)

Lable Image:
---------
I only have one class so the label image, as 0 is background and others are multi classes. If you have 2 classes, the label image pixel value should be 012 and so on:  

![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/data/aachen_000001_000019_gtFine_labelIds.png)

Steps to train you own dataset:  
-----------------  

      1. prepare train images and label images like above images  
	  
	    I have added below function call in my code according to my pictures, you have to change it according to your image name  
		 
	    #######################################################
	    find_replace(labelpath, "_leftImg8bit.png", "_gtFine_labelIds.png", labelpath); 
	    #######################################################
		  
      2. put label images and original images together
      
      3. generate the train.list file just like:  
	  
	    /home/user/Desktop/YOLO_train/leftImg8bit/train/aachen_resize/jena_000012_000019_leftImg8bit.png
	
	  4. start train
	  
	    ./darknet segmenter train [data_file path] cfg/segment.cfg [pretrain weights file I gave to you]  
	
If you want to see my Result Video, I have put it in: https://pan.baidu.com/s/1uJwFYLHEQ9DGFZ8RkGuagg, and the password is: ic3q

If you want to get how to change the code, see https://github.com/ArtyZe/yolo_segmentation/blob/master/Train_Details.md     

What I did to change Yolo for image segmentation, I have written a blog in: https://blog.csdn.net/Artyze/article/details/82721147

After I will do some work in semantic segmentation with yolo.

If you want to do something with Yolo with me, contact me with E-mail: Gaoyang917528@163.com.
  
