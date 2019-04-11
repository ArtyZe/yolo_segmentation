# yolo_segmentation
the code is to get segmentation image by darknet

In the process of my project, I have referenced nithi89/unet_darknet in some points

and nithilan has give me many important advices, thanks to him, and if you have interest you can visit his homepage.

This is my third version, I added dilation convolutional, and now it has not so perfect result, but i think it's good enough for me. 

I will continue to update afterwards, please stay tuned.

[The Commond to Run My Project]
=========
Train: just like what I have anwsered in Issues,

	./darknet segmenter train [data_file path] cfg/segment.cfg [pretrain weights file I gave to you] 

Test:

	./darknet segmenter test [data_file path] cfg/segment.cfg [weights file] [image path]

Merge two images:

	python Merge.py
	
and you will get the mask image named final.png

Test image:  
![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/orig.png)

Output image:
![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/pred.png)

[How to Train with Your Own Dataset ?]  
========  

The Way is so easy, you only need three files, for example with cityscape dataset:

Colorful Original Image:  

![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/zurich_000118_000019_leftImg8bit.png)

Lable Image(I only have one class so the label image with pixels 0101, as 0 is background and 1 is object

if you have 2 classes, the label image pixel value should be 012 and so on:  

![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/zurich_000118_000019_gtFine_instanceIds.png)

maybe you can't see the pixels with value 1 because it's close to 0, just see the image below(not for training, just watch as example):  

![Image text](https://github.com/ArtyZe/yolo_segmentation/blob/master/zurich_000118_000019_gtFine_instanceIds_1.png)

Steps to train you own dataset:  
-----------------  

      1. prepare train images and label images
		 (I have added 
				find_replace(labelpath, "_leftImg8bit.png", "_gtFine_instanceIds.png", labelpath); 
		  in my code according to my pictures, you have to change it according to your image name) like above images
		  
      2. put label images and original images together;
      
      3. generate the train.list file just like:
			/home/user/Desktop/YOLO_train/leftImg8bit/train/aachen_resize/jena_000012_000019_leftImg8bit.png
	
	  4. start train
	  
	./darknet segmenter train [data_file path] cfg/segment.cfg [pretrain weights file I gave to you]  
	
If you want to see my Result Video, I have put it in: https://pan.baidu.com/s/1uJwFYLHEQ9DGFZ8RkGuagg, and the password is: ic3q

What I did to change Yolo for image segmentation, I have written a blog in: https://blog.csdn.net/Artyze/article/details/82721147

After I will do some work in semantic segmentation with yolo.

If you want to get the cfg and weights file, or you want to do something with Yolo with me, contact me with E-mail: Gaoyang917528@163.com.
  
