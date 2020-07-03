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
color_dict = {3: [0, 0, 0], 11: [70, 70, 70], 23: [180, 130,  70], 4: [0, 0, 0], 21: [ 35, 142, 107], 17: [153, 153, 153], 27: [70,  0,  0], 19: [ 30, 170, 250], 15: [100, 100, 150], 20: [  0, 220, 220], 26: [142,   0,   0], 24: [ 60,  20, 220], 25: [  0,   0, 255], 33: [ 32,  11, 119], 7: [128,  64, 128], 5: [  0,  74, 111], 8: [232,  35, 244], 1: [0, 0, 0], 28: [100,  60,   0], 6: [81,  0, 81], 0: [0, 0, 0], 22: [152, 251, 152], 13: [153, 153, 190], 30: [110,   0,   0], 16: [ 90, 120, 150], 31: [100,  80,   0], 9: [160, 170, 250], 32: [230,   0,   0], 12: [156, 102, 102], 18: [153, 153, 153]}
def Merge(img,img1):
   height,width, c = img1.shape
   print(img1.shape)
   for i in range(0, height):
      for j in range(0, width):
         if img[i,j] not in color_dict.keys():
            color_dict[img[i,j]] = [0,0,0]
         for c in range(3):
            img1[i,j,c] = color_dict[img[i,j]][c] + img1[i,j,c]
   cv2.imwrite("final.png",img1)
 
im = cv2.imread("pred.png", 0)
im1 = cv2.imread("orig.png")
im1 = cv2.resize(im1, (1024,512))


Merge(im, im1)