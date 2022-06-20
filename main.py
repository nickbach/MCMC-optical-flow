#  Copyright [2020] [Jan Dorazil]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import numpy as np
import matplotlib.pyplot as plt
from mcmc_of import MCMCOpticalFlow
from generate_sample_images import generate_sample_images
from lucas_kanade import lucaskanade
import time

# Generate test images
N = 80  # rows
M = 80  # cols
F, G, U, V = generate_sample_images(N, M, magnitude=1)

U_LK,V_LK = lucaskanade(F,G,11)

# Run MCMC
num = 3000
skip = 1000
of = MCMCOpticalFlow(F, G)
start = time.time()
u, v, lamb, delt = of.run(num)


# Calculate the MMSE (minimum mean squared error) estimate (mean)
u_mmse = np.mean(u[:, skip-1:num-1], axis=1)
v_mmse = np.mean(v[:, skip-1:num-1], axis=1)
U_mmse = np.reshape(u_mmse, (N, M))
V_mmse = np.reshape(v_mmse, (N, M))

# Calculate the uncertainty
u_var = np.var(u[:, skip-1:num-1], axis=1)
v_var = np.var(v[:, skip-1:num-1], axis=1)
U_var = np.reshape(u_var, (N, M))
V_var = np.reshape(v_var, (N, M))
std = np.sqrt(U_var + V_var)    # Combine the variances somehow to get an idea of the overall uncertainty


# Calculate the endpoint error
ep_err = np.sqrt((U - U_mmse)**2 + (V - V_mmse)**2)

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
ax[0, 0].set_title('Frame 1')
ax[0, 0].imshow(F)
ax[0, 1].set_title('Frame 2')
ax[0, 1].imshow(G)
X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='xy')
ax[1, 0].set_title('MMSE estimate')
ax[1, 0].quiver(X[::5,::5], Y[::5,::5], U_mmse[::5,::5], V_mmse[::5,::5])
ax[1, 1].set_title('Ground truth')
ax[1, 1].quiver(X[::5,::5], Y[::5,::5], U[::5,::5], V[::5,::5])
ax[1, 2].set_title('Lucas-Kanade estimate')
ax[1, 2].quiver(X[::5,::5], Y[::5,::5], -U_LK[::5,::5], -V_LK[::5,::5])
ax[2, 0].set_title('Endpoint error')
ax[2, 0].imshow(ep_err)
ax[2, 1].set_title('Uncertainty')
ax[2, 1].imshow(std)
ax[0, 2].axis('off')
ax[2, 2].axis('off')

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].set_title('Histogram of $\lambda$')
# ax[0].hist(lamb[skip-1:num-1])
# ax[0].set_xlabel('$\lambda$')
# ax[0].set_ylabel('frequency')
# ax[1].set_title('Histogram of $\delta$')
# ax[1].set_xlabel('$\delta$')
# ax[1].set_ylabel('frequency')
# ax[1].hist(delt[skip-1:num-1])
plt.show()

LKE = np.sqrt(np.square(-U_LK-U) + np.square(-V_LK-V))
num = np.where(std>np.mean(std), 1, 0)
U_err = np.where(std>np.mean(std), U_mmse, U)
V_err = np.where(std>np.mean(std), V_mmse, V)
BE = np.sqrt(np.square(U_err-U) + np.square(V_err-V))
LK_err = np.sum(LKE)/(N*M)
MNSE_err = np.sum(BE)/np.sum(num)


print('LK Error:', LK_err)
print('MNSE Error:', MNSE_err)
