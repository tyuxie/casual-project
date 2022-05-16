import torch
import torch.nn as nn
import sys
def nll(X,Z,D,Y,alpha,beta,eta,gamma,estimator='LATE'):
    # print(X.type(), beta.type())
    X@beta
    phi = torch.sigmoid(X@beta)
    c1, c3 = phi[:,0], phi[:,1]
    if estimator == 'LATE':
        c0 = torch.tanh(X @ alpha)
    elif estimator == 'MLATE':
        c0 = torch.exp(X @ alpha)
    c5 = torch.exp(X@eta)
    if estimator == 'LATE':
        f0 = (c5*(2-c0)+c0-torch.sqrt(c0**2 * (c5-1)**2 + 4*c5)) / (2 * (c5-1))
        f1 = f0 + c0
    elif estimator == 'MLATE':
        f0 = (-(c0+1)*c5+torch.sqrt(c5**2*(c0-1)**2 + 4*c0*c5)) / (2*c0*(1-c5))
        f1 = f0 * c0
    p011 = (1-c1)*c3
    p001 = (1-c1)*(1-c3)
    p110 = 0
    p100 = 0
    p111 = f1*c1 + p110
    p010 = f0*c1 + p011
    p101 = 1-p001-p011-p111
    p000 = 1-p010-p100-p110

    d = torch.sigmoid(X@gamma)
    l = D*Y*Z*p111*d + (1-D)*Y*Z*p011*d + D*(1-Y)*Z*p101*d + (1-D)*(1-Y)*Z*p001*d + D*Y*(1-Z)*p110*(1-d) + (1-D)*Y*(1-Z)*p010*(1-d) + D*(1-Y)*(1-Z)*p100*(1-d) + (1-D)*(1-Y)*(1-Z)*p000*(1-d)
    return torch.mean(torch.log(l.clamp(1e-10)))

def MLE(X, Z, D ,Y, estimator='MLATE', dr=True):
    N, p = X.shape
    alpha = nn.Parameter(torch.rand(size=(p,))*2-1)
    beta = nn.Parameter(torch.rand(size=(p,2))*2-1) ## only phi1 and phi3
    eta = nn.Parameter(torch.rand(size=(p,))*2-1)
    gamma = nn.Parameter(torch.rand(size=(p,))*2-1)
    opt = torch.optim.Adam(params=(alpha, beta, eta, gamma), lr=1e-3, weight_decay=0)
    optloss = float('inf')
    for i in range(10000):
        opt.zero_grad()
        loss = -nll(X,Z,D,Y,alpha,beta,eta,gamma, estimator)
        if loss.item() < optloss:
            if abs(loss.item() - optloss) < 1e-6:
                break
            optloss = loss.item()
            mlealpha = alpha.detach().clone()
            mlebeta = beta.detach().clone()
            mleeta = eta.detach().clone()
            mlegamma = gamma.detach().clone()
        print('Iter {} | loss {:.04f}'.format(i+1, loss.item()))
        loss.backward()
        opt.step()


# data = pd.read_excel('401ksubs.xls', header=0, dtype=np.float32)
# data = np.array(data, dtype=np.float32)
# data = torch.from_numpy(data)
# torch.save(data, '401k.pt')
data = torch.load('401k.pt')
Z = data[:,0]
X = torch.cat((torch.ones((data.shape[0],1)), data[:, [1,2,3,5,9]]), dim=-1)
D = data[:,7]
Y = data[:,8]

N, p = X.shape
NR = 1
torch.manual_seed(24)
mlealphas = torch.zeros(size=(NR, p))
drualphas = torch.zeros(size=(NR, p))
minimums, optlosses = [], []
for i in range(NR):
    idxes = torch.multinomial(torch.ones(N), N, replacement=True)
    Xdata = X[idxes].clone()
    MLE(X,Z,D,Y)