import os
import cv2
import numpy as np
from pylab import imshow

def std(x):
    x_mean=np.mean(x)
    x_std=np.sqrt(np.mean((x-x_mean)**2))
    return x_std
    
class smp_patch:
    def __init__(self,img_path,patch_size,num_patch,upscale):
        self.img_path=img_path
        self.patch_size=patch_size
        self.num_patch=num_patch
        self.upscale=upscale
        
    def getFilename(self):
        img_name=os.listdir(self.img_path)
        img_num=len(img_name)
        #print img_name
        return img_name,img_num
    
    def getPic(self):
        INMs,INBs=self.getFilename()
        npix_img=[]
        Xh=[]
        Xl=[]
        imgH=[]
        imgL=[]
        for i in INMs:
            img_dir=os.path.join(self.img_path,i)
            img=cv2.imread(img_dir)
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
            L,W,H=new_img.shape
            npix_img.append(L*W)
        npix_img=np.asarray(npix_img)
        npatch_img=np.asarray(np.floor(npix_img*
            self.num_patch/np.sum(npix_img)),dtype=np.int32)
        
        for i in range(INBs):
            patch_num=npatch_img[i]
            img_name=INMs[i]
            img_dir=os.path.join(self.img_path,img_name)
            img=cv2.imread(img_dir)
            new_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
            new_img=new_img/255.
            
            h,l,imgl=self.sample_patches(new_img[:,:,0],patch_num)
            Xh.extend(h)
            Xl.extend(l)
            img=np.asarray(img)
            imgl=np.asarray(imgl)
            imgH.append(img)
            imgL.append(imgl)
        return Xh,Xl,imgH,imgL 
    def sample_patches(self,im,patch_num):
        imh=im
        nrow,ncol=imh.shape

        x=np.random.permutation(nrow-2*self.patch_size)+self.patch_size
        y=np.random.permutation(ncol-2*self.patch_size)+self.patch_size
        
        x,y=np.meshgrid(x,y)
        a,b=x.shape
        
        xrow=np.reshape(x,[1,a*b]).squeeze()
        ycol=np.reshape(y,[1,a*b]).squeeze()
       
        if patch_num<len(xrow):
            xrow=xrow[1:patch_num]
            ycol=ycol[1:patch_num]
        patch_num=len(xrow)
        
        
        iml=cv2.resize(imh,(ncol/self.upscale,nrow/self.upscale),interpolation=cv2.INTER_CUBIC)
        iml=cv2.resize(iml,(ncol,nrow),interpolation=cv2.INTER_CUBIC)
            

        H=[]
        L=[]
        
        for i in range(patch_num):
            row=xrow[i]
            col=ycol[i]
            Hpatch=imh[row:row+self.patch_size,col:col+self.patch_size]
            Lpatch=iml[row:row+self.patch_size,col:col+self.patch_size]
            
            Hpatch=np.reshape(Hpatch,[1,self.patch_size**2]).squeeze()
            Lpatch=np.reshape(Lpatch,[1,self.patch_size**2]).squeeze()

            HR=Hpatch.tolist()
            LR=Lpatch.tolist()
            hr_std=std(HR)
            if hr_std>=0.01:
                H.append(HR)
                L.append(LR)
        return H,L,iml
if __name__=='__main__':
    img_path='/media/lmg/KINGSTON/data/traindata/'
    patch_size=32
    num_patch=10000
    upscale=2
    pic_patches=smp_patch(img_path,patch_size,num_patch,upscale)
    Xh,Xl,h,l=pic_patches.getPic()
    