import os
for i in range(10,99):
    os.system("./darknet segmenter test cfg/maskyolo.data cfg/instance_segment.cfg backup/instance_segment_1%d000.weights orig4.png"%i)
    os.system("python Merge.py")
    os.system("cp final.png result/2/final%d.png"%i)