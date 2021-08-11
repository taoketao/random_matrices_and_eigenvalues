import numpy as np
import numpy.linalg as la
from time import sleep, ctime
import matplotlib as mpl
from matplotlib import pyplot as plt

''' -------------------- utilities -------------------- '''
def Round(x, decimals=4):
    x = (10**decimals * x).astype(int)/10**decimals
    return x

''' -------------------- settings -------------------- '''
Scaling=7; MatrixSize=5; Decimals=4; Iterations=50; Sleep=0.03; Printing=False; epsilon=1e-10


''' -------------------- parameters -------------------- '''
square_matrix_sizes=[4,6,8,10,15,25,50,100]

def getmatr0(): return np.random.randint(0,2,((Iterations,MatrixSize,MatrixSize)))
matr0kind, matr0kind_full = 'binary {0,1}', 'Ber(0.5) binary with probs={0:1/2, 1:1/2}'
def getmatr1(): return np.random.randint(-1,2,((Iterations,MatrixSize,MatrixSize)))
matr1kind, matr1kind_full = 'ternary {-1,0,1}', 'Ternary with probs={-1:1/3, 0:1/3, 1:1/3}'
def getmatr2(): return np.random.uniform(0,1,((Iterations,MatrixSize,MatrixSize)))
matr2kind, matr2kind_full = 'uniform(0,1)', 'Uniform(0,1) with uniform probability on the unit interval'

def get_matrices(): return [getmatr0(), getmatr2(), getmatr1()]
def get_matrix_info_brief(): return [matr0kind,matr2kind,matr1kind]
def get_matrix_info_full(): return [matr0kind_full,matr2kind_full,matr1kind_full]
number_matrix_types = 3


''' -------------------- overhead -------------------- '''
means, stdevs = [],[]
number_of_statistics = 2
plot_height = number_matrix_types * number_of_statistics 

''' -------------------- setup, populate, and beautify Matplotlib plots -------------------- '''
fig, axs = plt.subplots(plot_height , len(square_matrix_sizes), figsize=(14, 7)) 
mpl.rcParams.update({'font.size': 6})
fig.suptitle("The absolute eigenvalues of random matrices, averaged over "+str(Iterations)+\
        " iterations. \nRows: "+';  '.join(get_matrix_info_full())+ \
        " all equally sampled. \n Eigenvalue means are sorted by absolute value are in red and "+\
        "corresponding standard deviations are in blue. Columns: "\
        +str(square_matrix_sizes)+"-sized square matrices. "+ctime())
# subplot labels
for nth_matrix, MatrixSize in enumerate(square_matrix_sizes): 
    axs[0, nth_matrix].set_title(str(MatrixSize)+'x'+str(MatrixSize)+' matrix')
    axs[plot_height-1,nth_matrix].set_xticks([0,MatrixSize],[1,MatrixSize])
    for r in range(0,plot_height-1):
        axs[r,nth_matrix].set_xticks([],[])

for distr_kind_iter, distribution_kind in enumerate( get_matrix_info_brief()):
    axs[0+distr_kind_iter*2,0].set_ylabel('mean \n'+distribution_kind, rotation=0, labelpad=20)
    axs[1+distr_kind_iter*2,0].set_ylabel('stdev\n'+distribution_kind, rotation=0, labelpad=20)

#axs[0,0].set_ylabel('mean \n'+,      rotation=0, labelpad=20)
#axs[1,0].set_ylabel('stdev\nbinary {0,1}',      rotation=0, labelpad=20)
#axs[2,0].set_ylabel('mean \nternary {-1,0,1}',  rotation=0, labelpad=20)
#axs[3,0].set_ylabel('stdev\nternary {-1,0,1}',  rotation=0, labelpad=20)
#axs[4,0].set_ylabel('mean \nuniform(0,1)',      rotation=0, labelpad=20)
#axs[5,0].set_ylabel('stdev\nuniform(0,1)',      rotation=0, labelpad=20)


''' -------------------- computations -------------------- '''
for nth_matrix, MatrixSize in enumerate(square_matrix_sizes):
    random_matrices = get_matrices()
    for mi_, matr in enumerate(random_matrices):
        eigs = la.eigvals(matr)
        eigs = abs(eigs)
        eigs = Round(eigs, decimals=Decimals)
        eigs.sort(1)
        eigs = np.fliplr(eigs)
        means += [Round(eigs.mean(0))]
        stdevs += [Round(np.std(eigs,0))]

        axs[0+2*mi_, nth_matrix].plot( means[-1], marker=".", c='red')
        axs[1+2*mi_, nth_matrix].plot( stdevs[-1], marker=".", c='blue')

        if Printing:
            print(eigs)
            print(eigs.shape)
            print(np.count_nonzero(eigs), MatrixSize*Iterations,\
                    np.count_nonzero(eigs)/float(MatrixSize*Iterations))

            print('mean: ', Round(eigs.mean(0)))
            print('stdev:', Round(np.std(eigs,0)))
            input()

plt.show()
plt.close()
