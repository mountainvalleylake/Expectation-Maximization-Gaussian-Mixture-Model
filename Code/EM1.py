import numpy as np
from scipy.stats import multivariate_normal
import sklearn as sk
Data, real_avg, real_sd = None, None, None
x_size = 0
N = 0
y_size = 0
w, est_miu, est_sigma = None, None, None
variance_matrix = None
NW, P = None, None
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part0 = ((2* np.pi)**(x_size/2)) * (abs(np.linalg.det(cov)))**(1/2)
    #print("Part0 ",part0)
    part1 = 1 / part0
    part2 = abs((-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu)))
    #print("Part1 ",part1," Part2 ",part2)
    p = part1 * part2[0][0]
    return p


def Input_Initialize():
    print("Initialize necessary parameters")
    global real_avg,real_sd,Data,N,x_size,y_size
    N=int(input("Enter Dataset Size: "))
    x_size=int(input("Enter Feature Vector Size: "))
    y_size=int(input("Enter Output Class Size: "))
    print(N,x_size,y_size)
    np.random.seed(0)
    #Generate Dummy Value
    real_avg = np.random.randn(y_size,x_size)
    real_sd = np.random.randn(y_size, x_size, x_size)
    Data = np.random.randn(N,x_size)
    for i in range(y_size):# for each class
        for h in range(x_size):# for each feature vector
            miu = np.random.uniform(low=50,high=500)
            real_avg[i][h] = miu
    for i in range(y_size):
        real_sd[i] = real_sd[i] * np.transpose(real_sd[i])
        #print(real_sd[i])
    #print(real_sd)

def Data_Generation():
    print("Generate Data")
    global real_avg, real_sd, Data,N,x_size,y_size
    for i in range(N):
        z = np.random.randint(low=0,high=y_size) #pick up any random class
        #print(z)
        miu = real_avg[z]
        sigma = real_sd[z]
        d = np.random.multivariate_normal(miu,sigma,1)
        Data[i] = d
    #print(Data)

def Assume_Initial():
    global y_size,x_size,N,NW,P
    global w,est_miu,est_sigma
    print("Initialize W, Mean and SD")
    np.random.seed(0)
    #w = np.random.uniform(0, 1, (y_size, 1))
    w = [1. / y_size] * y_size
    NW = np.zeros((N,y_size))
    #est_miu = np.random.uniform(0, 1, (y_size, x_size)) * 100
    est_miu = Data[np.random.choice(N, y_size, False), :]
    #est_sigma = np.random.uniform(0, 1, (y_size, x_size, x_size)) * 100
    est_sigma = [np.eye(x_size)] * y_size
    #est_sigma = sk.datasets.make_spd_matrix(y_size,x_size,x_size)

def Calculate_Likelihood():
    global x_size, y_size, P, N, w, NW, est_miu, est_sigma, Data
    print("Likelihood")
    log_likelihood = np.sum(np.log(np.sum(NW, axis=1)))
    NW = (NW.T / np.sum(NW, axis=1)).T
    return log_likelihood

def Calculate_Expectation():
    print("Expectation")
    global NW, P, N, y_size, Data, est_miu, est_sigma, w
    Eqn = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-Data.shape[1] / 2.) \
                      * np.exp(-.5 * np.einsum('ij, ij -> i',
                                               Data - mu, np.dot(np.linalg.inv(s), (Data - mu).T).T))
    for i in range(y_size):
        NW[:, i] = w[i] * Eqn(est_miu[i], est_sigma[i])
        #print("Sum pji ",sum(P[j]))


def Calculate_Maximization():
    print("Maximization")
    global x_size, y_size, N,NW ,w, est_sigma, est_miu
    P = np.sum(NW, axis = 0)
    for i in range(y_size):
        # mean
        est_miu[i] = 1. / P[i] * np.sum(NW[:, i] * Data.T, axis=1).T
        x_mu = np.matrix(Data - est_miu[i])
        # covariance
        est_sigma[i] = np.array(1 / P[i] * np.dot(np.multiply(x_mu.T, NW[:, i]), x_mu))
        # probability
        w[i] = 1. / N * P[i]


def EM_Algorithm():
    print("EM Algorithm")
    epoch = 1000
    #old_value = -100000
    #new_value = old_value + 1
    #Assume w,miu,sigma by random number
    Assume_Initial()
    for itr in range(epoch):
        Calculate_Expectation()
        Calculate_Maximization()
        new_value = Calculate_Likelihood()
        print("Likelihood is ", new_value, " for epoch: ", itr)
        # if (new_value < old_value):
        #     print("Final Likelihood is ",old_value)
        #     break
        # old_value = new_value
    print(est_miu,real_avg)
    print(est_sigma,real_sd)

def main():
    Input_Initialize()
    Data_Generation()
    EM_Algorithm()


if __name__ == '__main__':
    main()