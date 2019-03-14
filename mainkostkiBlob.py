#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from matplotlib import pylab as plt
from matplotlib import image
import numpy as np
import cv2
import threading
import time
import logging 
import math
#global variables
buforImg=None         
buforGrupy=None
buforSr=None
koniec=0
frame = np.zeros((480,640,3),dtype = np.uint8) #resolution
aktualizacja=0
sem=threading.Lock()
sem2=threading.Lock()
kolory=[(255,255,),(255,0,0),(0,255,0),(0,0,255),(255,255,0)]

def wyswietl(img,grupy,sr):
    # iterate over groups of dots and draw them
    if grupy is not None:
        im_with_keypoints = cv2.drawKeypoints(img, [], np.array([]), (0,0,0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for (i, v) in enumerate(grupy):
            if i == 0:
                im_with_keypoints = cv2.drawKeypoints(img, grupy[i], np.array([]), kolory[i % len(kolory)],
                                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, grupy[i], np.array([]),
                                                      kolory[i % len(kolory)],
                                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.putText(im_with_keypoints, str(len(v)), (sr[i][0], sr[i][1] - 20), cv2.FONT_ITALIC, 1, kolory[i], 2, cv2.LINE_AA)
        img=im_with_keypoints
    cv2.imshow('Detekcja oczek',img)

def odleglosc(x1,y1,x2,y2):
    d=math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return d

def worker():
    global frame,koniec,aktualizacja,buforImg,buforGrupy,buforSr #zmienne globalne
    tmpSr=[]
    tmpGrupy=[]
    tmpImg=[]
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8#0.6  # .3
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.8#0.01  # .5
    detector = cv2.SimpleBlobDetector_create(params)
    #detector = cv2.SimpleBlobDetector(params)
    print(threading.currentThread().getName(), 'Starting')
    while(True):
        if koniec==1:
            break
        time.sleep(0.125)# sleep in order to give time to camera's thread to write into buffer
        if sem.acquire(False):#try to lock critical section
            #check whether there is something in the frame buffer
            if frame is not None:#(len(frame) != 0):
                #assign content of the frame buffer to the private buffer
                tmpImg=frame
            sem.release()#unlock critical section
        else:
            logging.info("worker couldn't enter critical section")
        #unless buffer 'tmpImg' is empty, we can process a video frame
        if len(tmpImg)>0:
            keypoints = detector.detect(tmpImg)
            tmpGrupy = []
            # classification of detected dots (blobs)
            for (i, v) in enumerate(keypoints):
                if (i == 0):  # first dot
                    tmpGrupy.append([])
                    indx = len(tmpGrupy)
                    tmpGrupy[indx - 1].append(v)
                else:  # next dots
                    success = False
                    j = 0
                    # iterate over all groups
                    while (j < len(tmpGrupy) and success == False):
                        #for each dot already classified to a certain group
                        for ob in tmpGrupy[j]:
                            d = odleglosc(ob.pt[0], ob.pt[1], v.pt[0], v.pt[1])
                            if (d <= 6*ob.size):  # classification condition - compare distance between two dots and a certain value
                                tmpGrupy[j].append(v);success = True;break
                        j = j + 1
                    # if success is False, then we haven't assigned a dot to any group - we create a new one.
                    if success == False:
                        tmpGrupy.append([])
                        indx = len(tmpGrupy)
                        tmpGrupy[indx - 1].append(v)
            #im_with_keypoints = cv2.drawKeypoints(tmpImg, [], np.array([]), (13, 65, 155),
             #                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            for (i, v) in enumerate(tmpGrupy):
                #calculate average coordinate x and y for all dots in a group                 
                sr=[0,0]
                for ob in v:
                    sr[0]=sr[0]+ob.pt[0]
                    sr[1]=sr[1]+ob.pt[1]
                sr[0]= int(sr[0]/len(v))
                sr[1] = int(sr[1] / len(v))
                tmpSr.append(sr)
            aktualizacja = 1
            buforImg=tmpImg
            buforGrupy=tmpGrupy
            buforSr=tmpSr
            tmpSr = []
            tmpGrupy = []
            tmpImg=[]

    print (threading.currentThread().getName(), 'Exiting')

def camera():
    global frame,koniec,aktualizacja,buforImg,buforGrupy,buforSr
    print(threading.currentThread().getName(), 'Starting')
    localImg=None
    localGrupy = None
    localSr = None
    cap=cv2.VideoCapture(1) #(0)
    while (True):
        ret, fra = cap.read()
        if (fra is None): print("konczymy");koniec = 1;break
        #critical section for shared memory
        if sem.acquire(False):
            frame = fra
            sem.release()#unlock
        else:
            logging.info("camera couldn't enter critical section")
        if aktualizacja==1:
            aktualizacja=0
            localImg=buforImg
            localGrupy = buforGrupy
            localSr = buforSr
        else:
            localImg=fra

        wyswietl(localImg,localGrupy,localSr)

        if cv2.waitKey(1) & 0xFF == ord('q'):#press 'q' on keyboard to exit the application
            koniec=1
            break

    cap.release()
    cv2.destroyAllWindows()
    print(threading.currentThread().getName(), 'Exiting')

logging.basicConfig(level=logging.INFO)
c = threading.Thread(name='my_service', target=camera)
w = threading.Thread(name='worker', target=worker)

c.start()
w.start()


#if __name__ == '__main__':
