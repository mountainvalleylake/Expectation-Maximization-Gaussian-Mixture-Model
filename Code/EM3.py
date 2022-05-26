import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal, chi2
from sklearn.datasets import make_spd_matrix
Data, real_avg, real_sd = None, None, None
x_size = 0
N = 0
y_size = 0
w, est_miu, est_sigma = None, None, None
variance_matrix = None
NW, P = None, None

def Input_Initialize():
    print("Initialize necessary parameters")
    global real_avg,real_sd,Data,N,x_size,y_size
    N=int(input("Enter Dataset Size: "))
    x_size=int(input("Enter Feature Vector Size: "))
    y_size=int(input("Enter Output Class Size: "))
    print(N,x_size,y_size)
    np.random.seed(5)
    #Generate Dummy Value
    real_avg = np.random.uniform(5, 100, (y_size, x_size))
    real_sd = []
    for i in range(y_size):
        sigma = make_spd_matrix(x_size)
        #sigma = sigma * np.transpose(sigma)
        real_sd.append(sigma)
    Data = []
    #print(real_sd)
    #print(real_avg)
    #print(np.shape(real_sd))


def Data_Generation():
    print("Generate Data")
    global real_avg, real_sd, Data,N,x_size,y_size
    #np.random.seed(5)
    for i in range(N):
        z = np.random.randint(low=0,high=y_size) #pick up any random class
        #print(z)
        miu = real_avg[z]
        sigma = real_sd[z]
        d = np.random.multivariate_normal(miu, sigma)
        Data.append(d)
    #print(np.shape(Data))

def Assume_Initial():
    global y_size,x_size,N,NW,P
    global w,est_miu,est_sigma
    print("Initialize W, Mean and SD")
    np.random.seed(5)
    w = np.zeros((y_size,1))
    for i in range(y_size):
        w[i] = 1/y_size
    NW = np.random.uniform(0, 10, (N, y_size))
    P = np.random.uniform(0, 1, (N, y_size))
    est_miu = np.random.uniform(5, 20, (y_size, x_size))
    est_sigma = []
    for i in range(y_size):
        sigma = make_spd_matrix(x_size)
        #sigma = sigma * np.transpose(sigma)
        est_sigma.append(sigma)
    #print(w)
    print("Estimated Sigma ",est_sigma)
    print("Estimated Avg ",est_miu)


def Calculate_Likelihood():
    global x_size, y_size, N, w, NW, est_miu, est_sigma, Data
    print("Likelihood")
    like = 0
    for j in range(N):
        x = Data[j]
        #x = np.reshape(x,(x_size,1))
        x = np.reshape(x, (1, x_size))
        #print(x)
        for i in range(y_size):
            miu = est_miu[i]
            #miu = np.reshape(miu,(x_size,1))
            sigma = est_sigma[i]
            #sigma = np.reshape(sigma,(x_size, x_size))
            #NW[j][i] = w[i][0] * pdf_multivariate_gauss(x, miu, sigma)
            #NW[j][i] = w[i][0] * multivariate_normal(miu, sigma, True).pdf(x)
            NW[j][i] = w[i][0] * multivariate_normal.pdf(x, miu, sigma,True) +  0.00000001
            #print(NW[j][i])
            #print(pdf_multivariate_gauss(x,miu,sigma))
            # NW[j][i] = multivariate_normal.pdf(x, mean=miu, cov=sigma)
    for j in range(N):
        val = 0
        for i in range(y_size):
            val += sum(NW[j])
        like += np.log2(val)
    print(like)
    return like


def Calculate_Expectation():
    print("Expectation")
    global NW, P, N, y_size
    for j in range(N):
        sum = 0
        for i in range(y_size):
            sum += NW[j][i]
        for i in range(y_size):
            P[j][i] = NW[j][i] / sum
        #print("Sum pji ",sum(P[j]))
    # for j in range(N):
    #     summation = 0
    #     for i in range(y_size):
    #         summation += P[j][i]
    #     for i in range(y_size):
    #         P[j][i] /= summation


def Calculate_Maximization():
    print("Maximization")
    global x_size, y_size, N, w, est_sigma, est_miu
    sump = np.zeros((y_size,1))
    for i in range(y_size):
        for j in range(N):
            sump[i] += P[j][i]
    sumc = np.zeros((x_size,1))
    summ = np.zeros((x_size,1))
    for j in range(N):
        for i in range(y_size):
            for h in range(x_size):
                sumc[h] += P[j][i] *(Data[j][h] - est_miu[i][h])**2
                summ[h] += P[j][i] * Data[j][h]
    for i in range(y_size):
        w[i] = sump[i] / N
        for h in range(x_size):
            est_miu[i][h] = summ[h]/sump[i]
            est_sigma[i][h] = sumc[h]/sump[i]

    print("Updated Miu ", est_miu)
    print("Updated Sigma ", est_sigma)

def EM_Algorithm():
    print("EM Algorithm")
    epoch = 100
    old_value = -np.inf
    #Assume w,miu,sigma by random number
    Assume_Initial()
    for itr in range(epoch):
        new_value = Calculate_Likelihood()
        print("Likelihood is ", new_value, " for epoch: ",itr)
        if (new_value <= old_value):
            print("Final Likelihood is ",old_value)
            break
        old_value = new_value
        Calculate_Expectation()
        Calculate_Maximization()
        #print(np.subtract(real_avg,est_miu))
    print("Final mean ", est_miu)
    print("Real mean ",real_avg)
    print("Final var ", est_sigma)
    print("Real Var ",real_sd)

def main():
    Input_Initialize()
    Data_Generation()
    EM_Algorithm()


if __name__ == '__main__':
    main()