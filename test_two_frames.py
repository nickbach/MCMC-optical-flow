import numpy as np
from skimage.io import imread, imsave
from os.path import normpath as fn
from scipy.signal import convolve2d
from lucas_kanade import lucaskanade
import matplotlib.pyplot as plt

n = 2
kernel = np.ones((n, n))

F = np.float32(imread(fn('../Videos/frame10.jpg')))
print(F.shape)
# F = np.mean(F, axis = 2)/255.
F_convolved = convolve2d(F, kernel, mode='valid')
F = F_convolved[::n, ::n] / n

G = np.float32(imread(fn('../Videos/frame11.jpg')))
# G = np.mean(G, axis = 2)/255.
G_convolved = convolve2d(G, kernel, mode='valid')
G = G_convolved[::n, ::n] / n

u,v = lucaskanade(F,G,11)
# print(u[500, 500])

# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')
# plt.quiver(x[::4,::4],y[::4,::4],u[::4,::4],-v[::4,::4],pivot='mid')

plt.show()
