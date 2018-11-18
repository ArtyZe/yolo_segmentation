#coding=utf-8
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# fileName : Merge.py
# comment  : Merge the Original Picture and Ouput Picture
# version  :
# author   : ArtyZe
# date     : 
#
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import cv2

def Merge(img,img1):
   height,width, c = img1.shape
   for c in range(c):
     for i in range(0, height):
        for j in range(0, width):
           if(img[i,j] >90):
            #here 90 is Threshold for heatmap
            #print im1[i,j]
            img1[i,j,1] = 100+img1[i,j,1]
			
   cv2.imwrite("final.png",img1)
   return img1
 
im = cv2.imread("pred.png",cv2.IMREAD_GRAYSCALE)
im1 = cv2.imread("orig.png")


Merge(im, im1)