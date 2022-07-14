import numpy as np
import cv2 as cv2
import os, errno
from scipy.optimize import curve_fit
import warnings

def indexing(idx,dim):
    x = idx % dim
    y = np.int((idx-x)/dim)
    return [x,y]


def travel(x, a, c):
    return 1 / (1 + np.exp((x-c)/(2*a)))

def findnorm(hs,dim,count):
    if count == 0:
        return 0
    mid = np.int(dim/2)
    totaldist = 0
    for i in range(len(hs)):
        if hs[i] != 0:
            [x,y] = indexing(i,dim)
            dist = np.sqrt( (x-mid)**2 + (y-mid)**2 )
            totaldist = dist + totaldist
    return totaldist/count



def findextent(hs,dim,count):
    if count == 0:
        return [0,0]
    mid = np.int(dim/2)
    maxextent = 0
    totaldist = 0
    for i in range(len(hs)):
        if hs[i] != 0:
            [x,y] = indexing(i,dim)
            dist = np.sqrt( (x-mid)**2 + (y-mid)**2 )
            totaldist = dist + totaldist
            if(dist > maxextent):
                maxextent = dist
    return [maxextent, totaldist/count]


def area_np(hul):
    my_img = np.zeros((64, 64), dtype = "uint8")
    pts = hul.reshape((-1,1,2))
    cv2.polylines(my_img,[pts],True,255)
    cv2.fillPoly(my_img,[pts],255)
    area = sum(my_img.reshape(64*64)/255)
   
    return area

def hullarea(img):
    _,contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,2)
    out = np.concatenate(contours)
    hullpoints = cv2.convexHull(out)
    p = np.squeeze(hullpoints)
    q = area_np(p)
    return q

def hullareafind(hs,dim,count):
    a = (np.resize(hs,dim*dim).reshape(dim,dim)).astype(np.uint8)
    a = a * 255
    area = hullarea(a)
    density = np.float32(count)/area
    return [area, density]          
               
            
def fitcurve(hs,dim,norm):
    a = hs.reshape((dim,dim))
    a = cv2.resize(a, dsize=(dim*8, dim*8),interpolation=cv2.INTER_NEAREST)
    mid = np.int((dim*8)/2)
    polarimg = cv2.linearPolar(a,(mid,mid),dim*8,cv2.WARP_FILL_OUTLIERS)
    xdata = np.arange(0,mid)
    ydata = np.zeros(mid)
    for xx in range(0,mid):
        ydata[xx] = np.mean(polarimg[:,xx])
    xdata = xdata/(norm*8)
    try:
        popt, pcov = curve_fit(travel, xdata, ydata)
        return [popt[0],popt[1], polarimg,ydata]
    except RuntimeError:
        print( "Error - curve_fit failed")
        return [np.nan,np.nan,polarimg,ydata]