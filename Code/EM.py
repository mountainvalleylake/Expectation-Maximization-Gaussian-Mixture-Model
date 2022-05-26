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


def pdf_multivariate_gauss(x, mu, cov):
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
    #real_avg = np.random.uniform(5, 100, (y_size, x_size))
    real_avg = []
    val = -10
    for i in range(y_size):
        x = []
        val += 5
        for h in range(x_size):
            x.append(val)
            # print(len(x))
        real_avg.append(x)
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
        sigma = make_spd_matrix(x_size)*30
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
            NW[j][i] = w[i][0] * multivariate_normal.pdf(x, miu, sigma,True) + 0.00000001
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
    sumr = np.zeros((N))
    for j in range(N):
        for i in range(y_size):
            sumr[j] += NW[j][i]
    for j in range(N):
        for i in range(y_size):
            P[j][i] = NW[j][i] / sumr[j]
        #print("Sum pji ",sum(P[j]))


def Calculate_Maximization():
    print("Maximization")
    global x_size, y_size, N, w, est_sigma, est_miu
    sump = np.zeros((y_size))
    for i in range(y_size):
        for j in range(N):
            sump[i] += P[j][i]
    for i in range(y_size):
        pijx = np.zeros((1, x_size))
        for j in range(N):
            x = Data[j]
            pijx += np.multiply(P[j][i],x)
        est_miu[i] = pijx / sump[i]
    for i in range(y_size):
        pijt = np.zeros((x_size, x_size))
        for j in range(N):
            x = Data[j]
            subt = np.matrix(x-est_miu[i])
            mult_sigma = P[j][i] * np.multiply(np.transpose(subt) , subt)
            #print("Every step : ",mult_sigma)
            pijt += mult_sigma
        est_sigma[i] = pijt / sump[i]
        #print("Pji sum ", pji)
    for i in range(y_size):
        w[i][0] = sump[i] / N
        #print("Upadated Weight ",w[i][0])
    # for i in range(y_size):
    #     est_sigma[i] = est_sigma[i] * np.transpose(est_sigma[i])
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
        for j in range(N):
            plt.scatter(Data[j][0], Data[j][1], s=1, color="blue")
        for i in range(y_size):
            drawEllipse(est_sigma[i], est_miu[i])
        plt.show()
        #print(np.subtract(real_avg,est_miu))
    print("Final mean ", est_miu)
    print("Real mean ",real_avg)
    print("Final var ", est_sigma)
    print("Real Var ",real_sd)
    #print(P)
    for j in range(N):
        plt.scatter(Data[j][0], Data[j][1], s=1, color="blue")
    for i in range(y_size):
        drawEllipse(est_sigma[i], est_miu[i])
        # plt.gca().get_lines()[i].set_color("black")
    #plt.show()
    #plt.show()
    pylab.savefig("plot1.png")

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def drawEllipse(sigma, mu, ax=None):
    if ax is None:
        ax = plt.gca()
    values, vectors = eigsorted(sigma)
    angle = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
    feaw = {'facecolor': 'none', 'edgecolor': [0, 0, 0], 'alpha': 1, 'linewidth': 2}
    w, h = 2 * np.sqrt(chi2.ppf(0.5, 2)) * np.sqrt(values)
    e = Ellipse(xy=mu, width=w, height=h, angle=angle, **feaw)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)


def main():
    Input_Initialize()
    Data_Generation()
    EM_Algorithm()


if __name__ == '__main__':
    main()