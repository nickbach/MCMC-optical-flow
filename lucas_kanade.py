## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    u = np.zeros(f1.shape)
    v = np.zeros(f1.shape)

    avg = (f1+f2)/2
    I_t = f1-f2
    I_x = conv2(avg, fx, 'same', 'symm')
    I_y = conv2(avg, fy, 'same', 'symm')
    I_x2 = I_x*I_x
    I_xy = I_x*I_y
    I_y2 = I_y*I_y
    I_xt = I_x*I_t
    I_yt = I_y*I_t

    kernel = np.ones((W, W))
    sumI_x2 = conv2(I_x2, kernel, 'same', 'fill')
    sumI_y2 = conv2(I_y2, kernel, 'same', 'fill')
    sumI_xy = conv2(I_xy, kernel, 'same', 'fill')
    sumI_xt = conv2(I_xt, kernel, 'same', 'fill')
    sumI_yt = conv2(I_yt, kernel, 'same', 'fill')


    A = np.array([[sumI_x2[:, :]+10**-8, sumI_xy[:, :]], [sumI_xy[:, :], sumI_y2[:, :]+10**-8]])
    B = np.array([sumI_xt[:, :], sumI_yt[:, :]])
    A = np.transpose(A, (2, 3, 0, 1))
    B = np.transpose(B, (1, 2, 0))

    f = np.linalg.solve(A, B)

    u[:, :] = f[:, :, 0]
    v[:, :] = f[:, :, 1]

    return u,v

########################## Support code below

# from skimage.io import imread, imsave
# from os.path import normpath as fn # Fixes window/linux path conventions
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
#
#
# f1 = np.float32(imread(fn('../Videos/frame10.jpg')))/255.
# f2 = np.float32(imread(fn('../Videos/frame11.jpg')))/255.
#
# u,v = lucaskanade(f1,f2,11)
#
#
# # Display quiver plot by downsampling
# x = np.arange(u.shape[1])
# y = np.arange(u.shape[0])
# x,y = np.meshgrid(x,y[::-1])
# plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')
# # plt.quiver(x[::4,::4],y[::4,::4],u[::4,::4],-v[::4,::4],pivot='mid')
#
# plt.show()
