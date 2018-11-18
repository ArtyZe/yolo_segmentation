# yolo_segmentation
the code is to get segmentation image by darknet

In the process of my project, I have referenced nithi89/unet_darknet in some points

and nithilan has give me many important advices, thanks to him, and if you have interest you can visit his homepage.

This is my third version, I added dilation convolutional, and now it has not so perfect result, but i think it's good enough for me. 

I will continue to update afterwards, please stay tuned.

Now I have gave my cfg file and weights file in:

	https://pan.baidu.com/s/1vibb9nlfIV3NvBreBSaJbA
  
and the password is:  ncfb

[The Commond to Run My Project]

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

If you want to see my Result Video, I have put it in: https://pan.baidu.com/s/1uJwFYLHEQ9DGFZ8RkGuagg, and the password is: ic3q

What I did to change Yolo for image segmentation, I have written a blog in: https://blog.csdn.net/Artyze/article/details/82721147

After I will do some work in semantic segmentation with yolo.

If you want to do something with Yolo with me, contact me with E-mail.
  


