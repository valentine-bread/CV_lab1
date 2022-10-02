#%%

import cv2, time
import numpy as np
from numba import njit, prange

@njit(fastmath = True, parallel = True)
def gen_gaussian_kernel_numba_2d(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = np.zeros((kernel_size, kernel_size))
    karnel_sum = 0
    for i in prange(kernel_size): 
        for j in prange(kernel_size): 
            x = np.exp(-((i - kernelRadius)**2 + (j - kernelRadius)**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
            karnel[i][j] = x
            karnel_sum += x
    for i in prange(kernel_size):
        for j in prange(kernel_size):
            karnel[i][j] = karnel[i][j] / karnel_sum
    return karnel

@njit(fastmath = True)
def gen_gaussian_kernel_numba(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = np.zeros(kernel_size)
    karnel_sum = 0
    for i in prange(kernel_size): 
        # x = np.exp(- (i - kernelRadius) ** 2 / (2 * sigma ** 2))
        x = np.exp(-((i - kernelRadius)**2)/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
        karnel[i] = x
        karnel_sum += x
    for i in prange(kernel_size):
        karnel[i] = karnel[i] / karnel_sum
    return karnel

def gen_gaussian_kernel(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    # karnel = np.array([np.exp(- (i - kernelRadius) ** 2 / (2 * sigma ** 2)) for i in prange(kernel_size)])   
    karnel = np.array([np.exp(-((i - kernelRadius)**2)/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2) for i in prange(kernel_size)])   
    karnel = karnel / karnel.sum() 
    return karnel

def GaussianBlur(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    karnel = gen_gaussian_kernel(kernel_size, sigma)
    kernelRadius = int((kernel_size - 1) / 2)
    height, width, channel = img.shape
    for channel_i in range(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    h = height_i - kernelRadius + i 
                    sum_pix += img_cv[abs(h) if h < height else 2*height - h - 1][width_i][channel_i] * karnel[i] 
                img[height_i][width_i][channel_i] = sum_pix
                
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for j in range(kernel_size):
                    w = width_i - kernelRadius + j 
                    sum_pix += img[height_i][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[j]
                img[height_i][width_i][channel_i] = sum_pix
    return img

@njit(fastmath = True, parallel = True)
def GaussianBlur_numba(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = gen_gaussian_kernel_numba(kernel_size, sigma)
    # print(karnel)
    # karnel = cv2.getGaussianKernel(kernel_size, sigma))
    # karnel = np.array([0,0,0,0,0,1,0,0,0,0,0])
    height, width, channel = img.shape
    for channel_i in prange(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    h = height_i - kernelRadius + i 
                    sum_pix += img_cv[abs(h) if h < height else 2*height - h - 1][width_i][channel_i] * karnel[i] 
                img[height_i][width_i][channel_i] = sum_pix
                
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for j in range(kernel_size):
                    w = width_i - kernelRadius + j 
                    sum_pix += img[height_i][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[j]
                img[height_i][width_i][channel_i] = sum_pix
    return img
    
@njit(fastmath = True, parallel = True)
def GaussianBlur_numba_2d(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = gen_gaussian_kernel_numba_2d(kernel_size, sigma)
    # print(karnel)
    # karnel = cv2.getGaussianKernel(kernel_size, 1))
    # karnel = np.array([0,0,0,0,0,1,0,0,0,0,0])
    height, width, channel = img.shape
    for channel_i in prange(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        h = height_i - kernelRadius + i 
                        w = width_i - kernelRadius + j 
                        sum_pix += img[abs(h) if h < height else 2*height - h - 1][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[i][j] 
                img[height_i][width_i][channel_i] = sum_pix
    return img

def GaussianBlur_cv2(img_cv, karel_size, sigma):
    return cv2.GaussianBlur(img_cv, (karel_size, karel_size),sigma)  

def gen_img(size):
    height, width = size
    blank_image = np.random.rand(height,width,3)
    blank_image = blank_image * 255
    return blank_image
img_cv = cv2.imread('img/3.png')    


#%%
GaussianBlur_numba(img_cv, 1, 1)
GaussianBlur_numba_2d(img_cv, 1, 1)
sigma = 4
function = [GaussianBlur_cv2, GaussianBlur_numba, GaussianBlur_numba_2d, GaussianBlur]
#%%
stat_k_size = dict((f.__name__, dict()) for f in function)
karel_size = range(1, 16, 2)
for f in function:
    for k_size in karel_size: 
        size = (1000, 1000)
        img = gen_img(size)
        start_time = time.time()  
        f(img, k_size, sigma)
        work_time = (time.time() - start_time) 
        stat_k_size[f.__name__][k_size] =  work_time
        print(f.__name__, size, k_size, work_time)
        
# %%  
stat_img_size = dict((f.__name__, dict()) for f in function)      
# img_size = zip(range(100, 2001, 100), range(100, 2001, 100))
img_size = [(x,x) for x in range(100, 2001, 100)]
for f in function:
    for size in img_size: 
        k_size = 11
        img = gen_img(size)
        start_time = time.time()  
        f(img, k_size, sigma)
        work_time = (time.time() - start_time) 
        stat_img_size[f.__name__][size] = work_time
        print(f.__name__, size, k_size, work_time)

# %%
import matplotlib.pyplot as plt
for f in function[:-1]:
    plt.plot(stat_k_size[f.__name__].keys(), stat_k_size[f.__name__].values(), label = f.__name__)
plt.xlabel('$karnel size$')
plt.ylabel('$time$')
plt.legend(loc='upper left')

# %%

for f in function[:-1]:  
    plt.plot(list(map(lambda x: x[0]*x[1], stat_img_size[f.__name__].keys())), stat_img_size[f.__name__].values(), label = f.__name__)
plt.xlabel('$image size$')
plt.ylabel('$time$')
plt.legend(loc='upper left')
# %%