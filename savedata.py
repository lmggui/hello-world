import numpy as np
import scipy.io as sio
import smp_patch
import cPickle


img_path='/media/lmg/anything/study/SR/data/'
patch_size=6
num_patch=40000
upscale=2
pic_patches=smp_patch.smp_patch(img_path,patch_size,num_patch,upscale)
Xh,Xl,l,h=pic_patches.getPic()
x_train = np.asarray(Xl[0:20000])
y_train = np.asarray(Xh[0:20000])
x_test=np.asarray(Xl[20000:22000])
y_test=np.asarray(Xh[20000:22000])

#sio.savemat('./data.mat',{'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test})


cPickle.dump({'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test},open('data.pkl','wb'))