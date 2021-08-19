# random_matrices_and_eigenvalues
Scripts, experiments, results, and reports on questions surrounding spectra of random matrices.

**below is draft material. please dont quote me on it! I need to fact-check several claims**

Recurrent neural network attractors have the potential to act as state systems, automata, and turing-complete computers. Therefore, a better understanding of these attractors could shed light on how neural networks in a cortex might possibly implement complex computation. Their fundamentally distributed nature also begs the question of the extent to which state systems can be distributed while still retaining computational expressiveness. First we seek to understand the basic properties of these attractors, such as How many are there, When during learning to they arise, Are they the same shape, and How difficult is it to move an instant set of activations from one attractor to a different?

These initial experiments study the shape and numerosity of attractors of a fully-learned RNN via eigenspectra. Recurrent neural network connections can be modeled by a graph or connectivity matrix; repeated application of a matrix operator correspond to recurrent activations over time. This scenario can be analyzed with tools from dynamical systems theory involving stationary distributions and ergodic theory. Verily, the term 'attractor' comes from dynamical systems, referring to states that a consistent system tends towards over time. As it turns out, the attractors given by a matrix correspond directly to its eigenvector spectrum. If an eigenvector has eigenvalue equal to 1, then it is an attractor. ((todo: insert latex inline proof)) For random strictly stochastic matrices, there is exactly one eigenvector with eigenvalue 1 (with all other eigenvectors admitting smaller eigenvalues) so there is one attractor. So by analyzing the eigenspectra of neural network connectivity matrices, we can begin to understand the true capabilities these attractors. 

In the code in script.py, I've modeled different kinds of connectivity matrices, each with specific purposes. Then, I analyzed the spectra of these matrices. The (absolute) strength of the the eigenvalues correspond to the basin size of an attractor, roughly speaking ((todo fix this up)), so the first analysis was geared towards looking at how quickly the eigenvalues decreased. The first matrices I analyzed were:
 - Binary matrices, where each element of the NxN matrix is drawn ~ Ber(0.5). This is a simple connectivity matrix. Its biological interpretation is that some neurons connect to each other and some don't in a way that seems random to us, but this is mostly as a sanity check.
 - Uniform matrices taking on real values between 0 and 1, "Unif(0,1)". It is also used as a point of comparison / sanity check.
 - Ternary matrices, with elements taking on values -1, 0, and 1 with equal probability. This is similar to a Hopfield network. Understanding the spectra of these matrices were a core part of this computer experiment.
 - uniform(-1,1)
 - uniform discrete of [0,0.5,1,1.5,2,2.5,3]
 - normal, mean 0, stdev 1
 - normal, mean 0.2, stdev 1
 - normal, mean -0.2, stdev 1
 - two-humped normal, norm(.2)+norm(-.2)
 - hopfield ternary, which is a ternary but with diagonals zeroed out
 - orthonormal from the orthogonal group
 - orthonormal with gaussian noise stutter
 - Identity
 

 not yet implemented:
 - binary {-1,1} matrices
 - Hopfield network-like binary matrices, which are like binary random matrices except the diagonals are zero
 - actual Hopfield net with all its properties, such as matrix symmetry
 - nearly orthonormal matrices 
 - partially orthogonal matrices

 - 'eigenvalues' of a neural network layer, found via simulation and sampling
   - MLP, RELU + Gaussian(0.01,1)/100.0, softmax + Uniform(0,N)/sqrt(N), RELU + np.random.randn(N) x sqrt(2./N) [He 2015 Delving]
     See https://cs231n.github.io/neural-networks-2/#init on initializations

For each type of matrix, I sampled many random instances, computed the eigenvalues, sorted them, and took the average (and stdev) over the instances. For plots made before 02:50 8.11, I used absolute value of eigenvalues, and after that, used singular values ie squared eigenvalues. See plots binary_ternary_uniform_01.png and ####XYZXYZ####. It's well known in dynamical systems theory that binary random matrices tend to have one large eigenvalue near unity and the remaining eigenvalues are about the same at value much less than one. The plots show as much: As the size of the square matrix increases, the binary matrix indeed seems to achieve one large eigenvaue and many smaller ones at a similar place. ((todo: are the remaining ones normally distributed?)). It also seems to be the case for Unif(0,1) matrices. But interestingly for ternary matrices, the eigenvalues degrade smoothly. This contrasts with the intuition we have about mode-learning of neural networks thanks to Saxe, Ganguli, and McClelland, 2013, "Exact Solutions...": we'd expect that networks would tend to learn one orthonormal eigenmode(?) at a time in correspondence with the only-one-large-eigenvalue.

anyways, there's math behind all this, and these experiments are for intuition mostly. Enjoy the script!


Update 8.18: turning this script stuff into a reproducible jupyter notebook.


Conclusive analysis so far: indeed, some of the distributions match the curvature of the Marchenko-Pastur distribution, as they should. Specifically, they are qualitatively similar to y=sqrt(4/x-1).
