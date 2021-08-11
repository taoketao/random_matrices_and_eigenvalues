import numpy as np
import numpy.linalg as la
from time import sleep, ctime
import sympy
import matplotlib as mpl
from matplotlib import pyplot as plt

def Round(x, decimals=4):
    x = (10**decimals * x).astype(int)/10**decimals
    return x

if 1:
    Scaling=7; MatrixSize=5; Decimals=4; Iterations=50; Sleep=0.03; Printing=False; epsilon=1e-10

    means, stdevs = [],[]
    sizes=[4,6,8,10,15,25,50,100]


    def getmatr0(): return np.random.randint(0,2,((Iterations,MatrixSize,MatrixSize)))
    def getmatr1(): return np.random.randint(-1,2,((Iterations,MatrixSize,MatrixSize)))
    def getmatr2(): return np.random.uniform(0,1,((Iterations,MatrixSize,MatrixSize)))

    def getmatrices():
        return [getmatr0(), getmatr1(), getmatr2()]
    number_matrix_types = 3

    number_of_statistics = 2
    plot_height = number_matrix_types * number_of_statistics 

    fig, axs = plt.subplots(plot_height , len(sizes), figsize=(14, 7)) 

    mpl.rcParams.update({'font.size': 6})
    fig.suptitle("The absolute eigenvalues of random matrices, averaged over "+str(Iterations)+\
            " iterations. Rows: Ber(p) binary, ternary RandInt({-1,0,1}), and Unif(0,1), "+\
            "all equally sampled. \n Eigenvalue means are sorted by absolute value are in red and "+\
            "corresponding standard deviations are in blue. Columns: "\
            +str(sizes)+"-sized square matrices. "+ctime())
    for m_, MatrixSize in enumerate(sizes):
        axs[0,m_].set_title(str(MatrixSize)+'x'+str(MatrixSize)+' matrix')
        axs[plot_height-1,m_].set_xticks([0,MatrixSize],[1,MatrixSize])
        for r in range(0,plot_height-1):
            axs[r,m_].set_xticks([],[])


    axs[0,0].set_ylabel('mean \nbinary {0,1}',      rotation=0, labelpad=20)
    axs[1,0].set_ylabel('stdev\n binary {0,1}',     rotation=0, labelpad=20)
    axs[2,0].set_ylabel('mean \nternary {-1,0,1}',  rotation=0, labelpad=20)
    axs[3,0].set_ylabel('stdev\nternary {-1,0,1}',  rotation=0, labelpad=20)
    axs[4,0].set_ylabel('mean \nuniform(0,1)',      rotation=0, labelpad=20)
    axs[5,0].set_ylabel('stdev\nuniform(0,1)',      rotation=0, labelpad=20)

    counter=0
    for m_, MatrixSize in enumerate(sizes):
        Data = np.zeros((MatrixSize, Iterations))
#        matr0 = getmatr0()
#        matr1 = getmatr0()
#        matr2 = getmatr0()
        random_matrices=getmatrices()
        for mi_, matr in enumerate(random_matrices):
            counter+=1
            eigs = la.eigvals(matr)

            eigs = abs(eigs)
            eigs = Round(eigs, decimals=Decimals)
            if Printing: print(type(eigs), eigs.dtype)
            eigs.sort(1)
            eigs = np.fliplr(eigs)

            if Printing:
                print(eigs)
                print(eigs.shape)
                print(np.count_nonzero(eigs), MatrixSize*Iterations,\
                        np.count_nonzero(eigs)/float(MatrixSize*Iterations))

                print('mean: ', Round(eigs.mean(0)))
                print('stdev:', Round(np.std(eigs,0)))
                input()
            means += [Round(eigs.mean(0))]
            stdevs += [Round(np.std(eigs,0))]

            axs[0+2*mi_, m_].plot( means[-1], marker=".", c='red')
            axs[1+2*mi_, m_].plot( stdevs[-1], marker=".", c='blue')

#            plt.sca(axs[0+2*mi_, m_])
#            plt.xticks([],[])
#            plt.sca(axs[1+2*mi_, m_])
#            plt.xticks([],[])

#            if mi_==len(random_matrices) and m_==len(sizes):
#            plt.sca(axs[0+2*mi_, m_])
#            plt.xticks([],[])
#            if counter==plot_height-1:
#                plt.sca(axs[0+2*mi_, m_])
##                axs[0+2*mi_, m_].set_xticks([0,1],[1,len(means)])
#                #plt.xticks([0,len(means)],[1,len(means)])
#                axs.set_xticks([0,len(means)],[1,len(means)])
#                plt.sca(axs[1+2*mi_, m_])
#                plt.xticks([0,len(stdevs)],[1,len(stdevs)])
#            else:
#                plt.sca(axs[0+2*mi_, m_])
#                plt.xticks([],[])

#            axs[0+2*mi_, m_].set_title('means of '+str(m_)+', '+ str(MatrixSize))
#            axs[1+2*mi_, m_].set_title('stdevs of '+str(m_)+', '+str(MatrixSize))
#    for mth_column, MatrixSize in enumerate(sizes):
#        axs[2, 1].set_xticks([0,len(means)],[1,len(means)])
##        plt.sca(axs[mth_column, plot_height-2])
##        plt.xticks([0,MatrixSize],[1,'alalaal'])
#        print( mth_column, plot_height -2 )
    plt.show()
    plt.close()
    import sys;sys.exit()




if 0:
    Scaling=7; MatrixSize=5; Decimals=4; Iterations=7; Sleep=0.03; Printing=False; epsilon=1e-10

    means, stdevs = [],[]
    sizes=[4,6,8,10,15,25,50,100]
    fig, axs = plt.subplots(2, len(sizes))#, sharex=True, sharey=True)

    for m_, MatrixSize in enumerate(sizes):
        Data = np.zeros((MatrixSize, Iterations))
        matr = np.random.randint(-1,2,((Iterations,MatrixSize,MatrixSize)))
     #   matr = np.random.randint(0,2,((Iterations,MatrixSize,MatrixSize)))
        matr = np.random.uniform(0,1,((Iterations,MatrixSize,MatrixSize)))
        eigs = la.eigvals(matr)

        eigs = abs(eigs)
        eigs = Round(eigs, decimals=Decimals)
        print(type(eigs), eigs.dtype)
        eigs.sort(1)
        eigs = np.fliplr(eigs)

        print(eigs)
        print(eigs.shape)
        print(np.count_nonzero(eigs), MatrixSize*Iterations,\
                np.count_nonzero(eigs)/float(MatrixSize*Iterations))

        print('mean: ', Round(eigs.mean(0)))
        print('stdev:', Round(np.std(eigs,0)))
        input()
        means += [Round(eigs.mean(0))]
        stdevs += [Round(np.std(eigs,0))]

        axs[0, m_].plot( means[-1], marker=">")
        axs[1, m_].plot( stdevs[-1], marker=">")

        axs[0, m_].set_title('means of '+str(m_)+', '+ str(MatrixSize))
        axs[1, m_].set_title('stdevs of '+str(m_)+', '+str(MatrixSize))
    plt.show()
    plt.close()
    import sys;sys.exit()

if 0:
  Scaling=7; MatrixSize=7; Decimals=4; Iterations=235; Sleep=0.03; Printing=False
  epsilon=1e-10

  Data = np.zeros((MatrixSize, Iterations))
  for t in range(Iterations):
    matr=np.random.randint(-1,2,(MatrixSize,MatrixSize))
    rounded_matr = np.round(matr, Decimals)
    matr = np.round(matr, Decimals)
#    eigs = la.eigvals(matr)*Scaling
    eigs = la.eigvals(matr)*Scaling
    lastdata=[]
    for i,eig in enumerate(sorted(list(abs(eigs)), reverse=True)):
        lastdata += [eig]
        logeig = eig/np.log2(MatrixSize)
        if Printing: print(str.ljust(str(round(logeig/Scaling,Decimals)),Decimals+2), '-'*int(i))
        Data[i,t] = eig

    if Printing: print('1    ','='*Scaling)
    if 1:
     det = 0 if la.det(matr)<epsilon else la.det(matr)
     try:
#        assert( (la.matrix_rank(matr, tol=epsilon)==MatrixSize) == bool(det) ) # sanity check singulars
        assert( (la.matrix_rank(matr)==MatrixSize) == bool(det) ) # sanity check singulars
     except:
        print('iter',t)
        print(Data)
        print(lastdata)
        print(det)
        print(sympy.Matrix(matr).rref())
#        print(sympy.Matrix(np.random.randint(-1,2,(MatrixSize,MatrixSize))).rref())
        print(la.matrix_rank(matr, tol=epsilon))
        print(MatrixSize)
        print(matr)
        print(bool(la.det(matr)))  # sanity check singulars
        assert(0)
     if Printing: print( 'singular' if la.matrix_rank(matr)<MatrixSize else 'nonsingular' )
    if Printing: print()

    sleep(Sleep)


if 0:
  Scaling=7; MatrixSize=7; Decimals=4; Iterations=35; Sleep=0.1
  for _ in range(Iterations):
    matr=np.random.randint(-1,2,(MatrixSize,MatrixSize))
#    rounded_matr = np.round(matr, Decimals)
    eigs = la.eigvals(matr)*Scaling
    for i in sorted(list(abs(eigs)), reverse=True):
        i/=np.log2(MatrixSize)
        print(str.ljust(str(round(i/Scaling,Decimals)),Decimals+2), '-'*int(i))
    print('1    ','='*Scaling)
#    print(la.matrix_rank(matr))
#    print(MatrixSize)
#    print(matr)
#    print(bool(la.det(matr)))  # sanity check singulars
    assert( (la.matrix_rank(matr)==MatrixSize) == bool(la.det(matr)) ) # sanity check singulars
    print( 'singular' if la.matrix_rank(matr)<MatrixSize else 'nonsingular' )
    print()

    sleep(Sleep)


if 0:
  Scaling=7; MatrixSize=7; Decimals=5; Iterations=5
  for _ in range(Iterations):
    matr = la.eigvals(np.random.randint(-1,2,(MatrixSize,MatrixSize)))*Scaling
    for i in sorted(list(abs(matr)), reverse=True):
        i/=np.log2(MatrixSize)
        print(str.ljust(str(round(i/Scaling,Decimals)),Decimals+2), '-'*int(i))
    print('1    ','='*Scaling)
    sleep(0.3)
