import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
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
    np.random.seed(5)
    #Generate Dummy Value
    real_avg = []#np.random.randint(5, 100, (y_size, x_size))
    val = -10
    for i in range(y_size):
        x = []
        val += 5
        for h in range(x_size):
            x.append(val)
            #print(len(x))
        real_avg.append(x)
    #real_avg = np.random.uniform(5, 100, (y_size, x_size))
    real_sd = []
    for i in range(y_size):
        sigma = make_spd_matrix(x_size)
        #sigma = sigma * np.transpose(sigma)
        real_sd.append(sigma)
    Data = []
    print(real_sd)
    print(real_avg)
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
    NW = np.zeros((N, y_size))
    P = np.zeros((N, y_size))
    #est_miu = np.random.uniform(10, 50, (y_size, x_size))
    est_miu = np.random.uniform(-10, 10, (y_size, x_size))
    #est_sigma = np.random.randint(1, 10, (y_size, x_size, x_size))
    #X = np.transpose(Data)
    est_sigma = []
    for i in range(y_size):
         sigma =  make_spd_matrix(x_size)
         #sigma = est_sigma[i]
         #est_sigma[i] = est_sigma[i] * np.transpose(est_sigma[i])
         #sigma = np.cov(X)
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
            NW[j][i] = multivariate_normal.pdf(x, miu, sigma, True) * w[i][0]
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
    for i in range(y_size):
        pji = 0
        pijx = np.random.randn(1,x_size)
        pijt = np.random.randn(x_size,x_size)
        pijx.fill(0)
        pijt.fill(0)
        for j in range(N):
            x = Data[j]
            pji += P[j][i]
            mult_miu = P[j][i] * x
            pijx = np.add(pijx,mult_miu)
        est_miu[i] = pijx / pji
        for j in range(N):
            x = Data[j]
            subt = np.subtract(x,est_miu[i])
            mult_sigma = P[j][i] * np.dot(subt,np.transpose(subt))
            #print("Every step : ",mult_sigma)
            pijt = np.add(pijt,mult_sigma)
        est_sigma[i] = pijt / pji
        #print("Pji sum ", pji)
        #print("Updated Miu ", est_miu[i])
        print("Updated Sigma ",est_sigma[i])
        w[i][0] = pji/N
        #print("Upadated Weight ",w[i][0])
    #for i in range(y_size):
    #    est_sigma[i] = est_sigma[i] * np.transpose(est_sigma[i])


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
    plot_gmm(est_miu, est_sigma, w, Data)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (x_size, x_size):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(y_size):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(est_miu, est_sigma, W, Data, label=True, ax=None):
    ax = ax or plt.gca()
    labels = P
    m = []
    n = []
    for i in range(N):
        m.append(Data[i][0])
        n.append(Data[i][1])
    if label:
         ax.scatter(m, n, c=labels, s=1, cmap='viridis', zorder=2)
    else:
        ax.scatter(m, n, s=1, zorder=2)
        ax.axis('equal')

    w_factor = 0.2 / max(W)
    for pos, covar, w in zip(est_miu, est_sigma, W):
        print("pos,cov,w", pos,covar,w)
        draw_ellipse(pos, covar, alpha=w * w_factor)
    #plt.show()
    pylab.savefig("plt2.png")


def main():
    Input_Initialize()
    Data_Generation()
    EM_Algorithm()


if __name__ == '__main__':
    main()