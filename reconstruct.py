import os
import cv2
import numpy as np
from pylab import figure,imshow
class reconstruct:
    def __init__(self,patch_size,scale,upscale):
        self.scale=scale
        self.patch_size=patch_size
        self.upscale=upscale
    def Hr2Lr(self,img):
        
        imh=img
        #cv2.imshow('imgh',im)
        nrow,ncol,ndim=imh.shape
        iml=cv2.resize(imh,(ncol/self.upscale,nrow/self.upscale),interpolation=cv2.INTER_CUBIC)
        iml=cv2.resize(iml,(ncol,nrow),interpolation=cv2.INTER_CUBIC)
        return iml    
    def get_patch(self,img):
        
        row,col=img.shape
        rows=np.arange(0,row-self.patch_size,self.patch_size-self.scale)
        cols=np.arange(0,col-self.patch_size,self.patch_size-self.scale)
        
        rows=np.append(rows,row-self.patch_size)
        cols=np.append(cols,col-self.patch_size)
        x,y=np.meshgrid(rows,cols)
        a,b=x.shape
        xrow=np.reshape(x,[1,a*b]).squeeze()
        ycol=np.reshape(y,[1,a*b]).squeeze()
        patch_num=len(xrow)
        PATCH=[]
        for i in xrange(patch_num):
            row=xrow[i]
            col=ycol[i]
            patch=img[row:row+self.patch_size,col:col+self.patch_size]
            patch=np.reshape(patch,[1,self.patch_size**2]).squeeze()
            pt=patch.tolist()
        
            PATCH.append(pt)
        return PATCH,xrow,ycol

    #def construct(self,img,imgl,Hp,row,col):
    def construct(self,img,Hp,row,col):
        r,l,b=img.shape
        hr=np.zeros([r,l])
        cntMat=np.zeros([r,l])
        patch_num=len(row)
        for i in xrange(patch_num):
            x=row[i]
            y=col[i]
            patch=Hp[:,i]
            hpatch=np.reshape(patch,[self.patch_size,self.patch_size])
            hr[x:x+self.patch_size,y:y+self.patch_size]+=hpatch
            cntMat[x:x+self.patch_size,y:y+self.patch_size]+=1
        hr=hr+0.000001
        hr=hr/cntMat
        
        #imgh=imgl+hr
        return hr

if __name__=='__main__':
    I=reconstruct(6,2,2)
    img=cv2.imread('F:/study/SR/data/penguin.jpg')
    new_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    
    imgl=I.Hr2Lr(img)
    lp,row,col=I.get_patch(new_img[:,:,0])
    lp=np.asarray(lp)
    hr=I.construct(img,lp.T,row,col)
    new_img[:,:,0]=hr
    imshow(new_img)
    img_rgb=cv2.cvtColor(new_img,cv2.COLOR_YCR_CB2RGB)
    figure()
    imshow(img_rgb)