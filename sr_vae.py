import numpy as np
import time
import os
from VAE import VAE
import cPickle
import gzip
import smp_patch

img_path='/media/lmg/anything/study/SR/data/'
patch_size=6
num_patch=20000
upscale=2
pic_patches=smp_patch.smp_patch(img_path,patch_size,num_patch,upscale)
Xh,Xl,l,h=pic_patches.getPic()

x_train = np.asarray(Xh[0:10000])
x_test=np.asarray(Xh[10000:10500])
hu_encoder = 200
hu_decoder = 200
n_latent = 100
continuous = True
n_epochs = 2000



print "instantiating model"
model = VAE(continuous, hu_encoder, hu_decoder, n_latent, x_train)

path = "./"
batch_order = np.arange(int(model.N / model.batch_size))
epoch = 0
LB_list = []



if __name__ == "__main__":
    print "iterating"
    start = time.time()
    while epoch < n_epochs:
        epoch += 1
#        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            batch_LB = model.update(batch, epoch)
            LB += batch_LB

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        if epoch%20==0 and epoch!=0:
            print "Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start)
            start=time.time()
#            np.save(path + "LB_list.npy", LB_list)
#        model.save_parameters(path)

    valid_LB = model.likelihood(x_test)
    print "LB on validation set: {0}".format(valid_LB)
    re_x=model.reconstruct(x_test)
    
    
    from scipy.ndimage import filters
    def backprojection(im_h,im_l,maxiter):
        row_l,col_l=im_l.shape
        row_h,col_h=im_h.shape
        im_h=np.double(im_h)
        im_l=np.double(im_l)
        for i in xrange(maxiter):
            im_l_s=cv2.resize(im_h,(col_l,row_l),interpolation=cv2.INTER_CUBIC)
            im_diff=im_l-im_l_s
            im_diff=cv2.resize(im_diff,(col_h,row_h),interpolation=cv2.INTER_CUBIC)
            im2=filters.gaussian_filter(im_diff,1)
            im_h=im_h+im2
        return np.round(im_h)    
    import cv2
    from pylab import figure,imshow,gray,axis
    import reconstruct
    scale=4
    img=cv2.imread('/media/lmg/KINGSTON/SRCNN_v1/SRCNN/Set14/lenna.bmp')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    def getlocal(img,x,y,size):
#        local=img[y-size/2:y+size/2-1,x-size/2:x+size/2]
#        return local
    #img=getlocal(img,100,100,100)
    
    constructImg=reconstruct.reconstruct(patch_size,scale,upscale)
    lImg=constructImg.Hr2Lr(img)
    figure()
    gray()
    axis('off')
    imshow(lImg)
    new_img=cv2.cvtColor(lImg,cv2.COLOR_RGB2YCR_CB)
    nrow,ncol,h=new_img.shape
    Lpatch,row,col=constructImg.get_patch(new_img[:,:,0])
    lp=np.asarray(Lpatch)
    N,diml=lp.shape

    hp=model.reconstruct(lp/255.)

    hImg=constructImg.construct(img,hp.T,row,col)
    
    im_l=cv2.resize(new_img[:,:,0],(ncol/upscale,nrow/upscale),interpolation=cv2.INTER_CUBIC)
    im_h=np.round(hImg*255)
    IM_H=backprojection(im_h,im_l,0)
    new_img[:,:,0]=IM_H
    new_img=cv2.cvtColor(new_img,cv2.COLOR_YCR_CB2RGB)
    figure()
    gray()
    axis('off')
    imshow(new_img)
    
    figure()
    axis('off')
    imshow(img)
    def psnr(im1,im2):
        if im1.shape==3:
            im1=cv2.cvtColor(im1,cv2.COLOR_RGB2YCR_CB)
            im1=im1[:,:,0].squeeze()
            im2=cv2.cvtColor(im2,cv2.COLOR_RGB2YCR_CB)
            im2=im2[:,:,0].squeeze()

        im1=np.double(im1)
        im2=np.double(im2)
#        im1=im1.squeeze()
#        im2=im2.squeeze()
        mse=((im1-im2)**2).mean()
        psnr=20*np.log10(255/np.sqrt(mse))
        return psnr
    rate1=psnr(img,lImg)
    rate2=psnr(img,new_img)
    a=66
    from pylab import imshow,subplot,gray
    subplot(121)
    imshow(re_x[a].reshape([6,6]),vmin=0,vmax=1)
    gray()
    subplot(122)
    imshow(x_test[a].reshape([6,6]),vmin=0,vmax=1)
