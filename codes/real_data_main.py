import torch
import torch.nn as nn
import sys
import time
def nll(X,Z,D,Y,alpha,beta,eta,gamma,estimator='LATE'):
    # print(X.type(), beta.type())
    phi = torch.sigmoid(X@beta)
    c1, c3 = phi[:,0], phi[:,1]
    if estimator == 'MLATE':
        c0 = torch.exp(X @ alpha)
    c5 = torch.exp(X@eta)
    # c5 = c5 * (torch.abs(c5-1) > 1e-10) + (torch.abs(c5-1) <= 1e-10) * (c5-1 >= 0.0) * (1+1e-10) + (torch.abs(c5-1) <= 1e-10) * (c5-1< 0.0) * (1-1e-10)
    # c5 = 1 + c5 - c5.clamp(1-1e-5,1+1e-5)
    c0 = c0.clamp(1e-10)
    if estimator == 'MLATE':
        f0 = torch.where(torch.abs(c5-1)>1e-10, (-(c0+1)*c5+torch.sqrt(c5**2*(c0-1)**2 + 4*c0*c5)) / (2*c0*(1-c5)), -(-c0-1+(2*(c0-1)**2+4*c0)/(c0+1)/2)/(2*c0))
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
    l = D*Y*Z*p111*d + (1-D)*Y*Z*p011*d + D*(1-Y)*Z*p101*d + (1-D)*(1-Y)*Z*p001*d + (1-D)*Y*(1-Z)*p010*(1-d) + (1-D)*(1-Y)*(1-Z)*p000*(1-d)
    # if torch.mean(torch.log(l.clamp(1e-10, 1))) :
    #     print(l)
    #     sys.exit(0)
    return torch.mean(torch.log(l.clamp(1e-10)))

def square_loss(X, Z, D, Y, alpha, beta, gamma, eta, estimator, strategy='identity'):
    d = torch.sigmoid(X@gamma)
    phi = torch.sigmoid(X@beta)
    phi1, phi3 = phi[:,0], phi[:,1]
    OP = torch.exp(X@eta)
    f = (d**Z) * ((1-d)**(1-Z))
    if estimator == 'MLATE':
        theta = torch.exp(X@alpha)
        H = Y * theta**(-D)
        f0 = torch.where(torch.abs(OP-1)>1e-10, (-(theta+1)*OP+torch.sqrt(OP**2*(theta-1)**2+4*theta*OP)) / (2*theta*(1-OP)), -(-theta-1+(2*(theta-1)**2+4*theta)/(theta+1)/2)/(2*theta))
        f1 = f0 * theta
        E = f0*phi1 +(1-phi1)*phi3
    
    if strategy == 'identity':
        return torch.sum((torch.sum(X*((2*Z-1)*(H-E)/f).unsqueeze(1), dim=0))**2)
    elif strategy == 'optimal':
        p011 = (1-phi1)*phi3
        p001 = (1-phi1)*(1-phi3)
        p110 = 0
        p100 = 0
        p111 = f1*phi1 + p110
        p010 = f0*phi1 + p011
        p101 = 1-p001-p011-p111
        p000 = 1-p010-p100-p110
        if estimator == 'MLATE':
            EH2_1 = p111/theta**2+p101
            EH2_0 = p110/theta**2+p100
            EH_1 = p111/theta+p101
            EH_0 = p110/theta+p100
            EZX = (EH2_1-EH_1**2) / d + (EH2_0-EH_0**2)/(1-d)
            w = -X * (1 / theta * f1 * phi1 / EZX).unsqueeze(1)
            return torch.mean((torch.mean(w*((2*Z-1)*(H-E)/f).unsqueeze(1), dim=0))**2)

def MLE(X, Z, D ,Y, estimator='MLATE', dr=True):
    N, p = X.shape
    alpha = nn.Parameter(torch.rand(size=(p,))*0.2-0.1)
    beta = nn.Parameter(torch.rand(size=(p,2))*0.2-0.1) ## only phi1 and phi3
    eta = nn.Parameter(torch.rand(size=(p,))*2-1)
    gamma = nn.Parameter(torch.rand(size=(p,))*0.2-0.1)
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
        if i % 100 ==0: 
            print('Iter {} | loss {:.04f}'.format(i+1, loss.item()))
        loss.backward()
        opt.step()
        if torch.sum(loss.isnan()) > 0:
            break
    if not dr:
        return mlealpha, mlebeta, mleeta, mlegamma
        
    alpha = nn.Parameter(mlealpha.clone(),requires_grad=True)
    # alpha = nn.Parameter(torch.rand(size=(2,))*2-1)
    opt = torch.optim.Adam(params=(alpha,), lr=1e-3, weight_decay=0)
    sqoptloss = float('inf')
    for i in range(10000):
        opt.zero_grad()
        sq_loss = square_loss(X, Z, D, Y, alpha, mlebeta, mlegamma, mleeta, estimator, strategy='optimal')
        if i % 100 ==0:
            print('Iter {} | sq_loss {:.08f}'.format(i+1, sq_loss.item()))
        if sq_loss.item() < sqoptloss:
            sqoptloss = sq_loss.item()
            drwalpha = alpha.detach().clone()
            if abs(sqoptloss) < 1e-6:
                break
        sq_loss.backward()
        opt.step()
        if torch.sum(sq_loss.isnan()) > 0:
            break
    return mlealpha, drwalpha, mlebeta, mleeta, mlegamma

data = torch.load('401k.pt')
data[:,1] /= 100
data[:,4] /= 10
data[:,9] /= 10000
Z = data[:,0]
X = torch.cat((torch.ones((data.shape[0],1)), data[:, [1,2,4,5,9]]), dim=-1)
D = data[:,7]
Y = data[:,8]

N, p = X.shape
NR = 100
rdn_seed = 5
torch.manual_seed(rdn_seed)
mlealphas = torch.zeros(size=(NR, p))
drwalphas = torch.zeros(size=(NR, p))
minimums, optlosses = [], []
for i in range(NR):
    t = time.time()
    print('Bootstrap {}'.format(i+1), '-'*50)
    idxes = torch.multinomial(torch.ones(N), N, replacement=True)
    Xdata, Zdata, Ddata, Ydata = X[idxes].clone(), Z[idxes].clone(), D[idxes].clone(), Y[idxes].clone()
    mlealpha, drwalpha, mlebeta, mleeta, mlegamma = MLE(Xdata, Zdata, Ddata, Ydata)
    mlealphas[i] = mlealpha
    drwalphas[i] = drwalpha
    print('time used: ', time.time()-t)

torch.save(mlealphas, 'real_mlealpha{}.pt'.format(rdn_seed))
torch.save(drwalphas, 'real_drwalpha{}.pt'.format(rdn_seed))