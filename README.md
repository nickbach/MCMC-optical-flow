Exploration into the efficacy of Bayesian optical flow was done in order to replace the need for techniques like corner detection in environments with low certainty, and to hopefully out perform them by allowing all pixels to be used with different weights instead of only corners. Detailed analysis shown in report.pdf.

Original README copied below:

This is essentialy a Bayesian implementation of the classical Horn-Schunck optical flow algorithm. The optical flow inference is formulated as an inverse problem which is solved via Gibbs sampling. It is based primarily on [1] however there are some slight changes to improve efficiency. In particular the optical flow is split into the vertical (u) and horizontal (v) component and each is sampled from its conditional pdf separately. Furthermore a more efficient method is used for sampling from a high-dimensional Gaussian distribution [2].

![Optical flow samples from the Gibbs sampler](sampling_convergence.gif "Optical flow samples from the Gibbs sampler")

[1] Sun, J., Quevedo, F. J. and Bollt, E. (2018) ‘Bayesian Optical Flow with Uncertainty Quantification’, Inverse Problems. Available at: https://arxiv.org/abs/1611.01230.

[2] F. Orieux, O. Féron, and J. Giovannelli, “Sampling high-dimensional Gaussian distributions for general linear inverse problems,” IEEE Signal Process. Lett., vol. 19, no. 5, p. 251, 2012
