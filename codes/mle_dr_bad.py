import torch
import torch.nn as nn
import sys
def generate(X, alpha, beta, eta, gamma, estimator="LATE"):
    phi = torch.sigmoid(X@beta)
    c1, c2, c3, c4 = phi[:,0], phi[:,1], phi[:,2], phi[:,3]
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
    p011 = (1-c1)*(1-c2)*c3
    p001 = (1-c1)*(1-c2)*(1-c3)
    p110 = (1-c1)*c2*c4
    p100 = (1-c1)*c2*(1-c4)
    p111 = f1*c1 + p110
    p010 = f0*c1 + p011
    p101 = 1-p001-p011-p111
    p000 = 1-p010-p100-p110
    VIden = torch.sigmoid(X @ gamma)

    Z = torch.bernoulli(VIden)

    p1 = torch.column_stack((p001, p101, p011, p111))
    p0 = torch.column_stack((p000, p100, p010, p110))
    p = Z.unsqueeze(1) * p1 + (1-Z).unsqueeze(1) * p0
    idx = torch.multinomial(p, num_samples=1).squeeze(1)
    cand = torch.LongTensor([[0,0], [1,0], [0,1], [1,1]])
    DY = cand[idx]
    return Z, DY[:,0], DY[:,1]

def nll(X,Xpl, Xpr,Z,D,Y,alpha,beta,eta,gamma,estimator='LATE'):
    phi = torch.sigmoid(Xpr@beta)
    c1, c2, c3, c4 = phi[:,0], phi[:,1], phi[:,2], phi[:,3]
    if estimator == 'LATE':
        c0 = torch.tanh(X @ alpha)
    elif estimator == 'MLATE':
        c0 = torch.exp(X @ alpha)
    c5 = torch.exp(Xpr@eta)
    if estimator == 'LATE':
        f0 = (c5*(2-c0)+c0-torch.sqrt(c0**2 * (c5-1)**2 + 4*c5)) / (2 * (c5-1))
        f1 = f0 + c0
    elif estimator == 'MLATE':
        f0 = (-(c0+1)*c5+torch.sqrt(c5**2*(c0-1)**2 + 4*c0*c5)) / (2*c0*(1-c5))
        f1 = f0 * c0
    p011 = (1-c1)*(1-c2)*c3
    p001 = (1-c1)*(1-c2)*(1-c3)
    p110 = (1-c1)*c2*c4
    p100 = (1-c1)*c2*(1-c4)
    p111 = f1*c1 + p110
    p010 = f0*c1 + p011
    p101 = 1-p001-p011-p111
    p000 = 1-p010-p100-p110

    d = torch.sigmoid(Xpl@gamma)
    l = D*Y*Z*p111*d + (1-D)*Y*Z*p011*d + D*(1-Y)*Z*p101*d + (1-D)*(1-Y)*Z*p001*d + D*Y*(1-Z)*p110*(1-d) + (1-D)*Y*(1-Z)*p010*(1-d) + D*(1-Y)*(1-Z)*p100*(1-d) + (1-D)*(1-Y)*(1-Z)*p000*(1-d)
    return torch.mean(torch.log(l.clamp(1e-10)))

def square_loss(X,Xpl,Xpr, Z, D, Y, alpha, beta, gamma, eta, estimator, strategy='identity'):
    d = torch.sigmoid(Xpl@gamma)
    phi = torch.sigmoid(Xpr@beta)
    phi1, phi2, phi3, phi4 = phi[:,0], phi[:,1], phi[:,2], phi[:,3]
    OP = torch.exp(Xpr@eta)
    f = (d**Z) * ((1-d)**(1-Z))
    if estimator == 'LATE':
        theta = torch.tanh(X@alpha)
        H = Y - D * theta
        f0 = (OP*(2-theta)+theta-torch.sqrt(theta**2*(OP-1)**2+4*OP))/(2*(OP-1))
        f1 = f0 + theta
        E = f0*phi1 + (1-phi1)*(1-phi2)*phi3 + (1-phi1)*phi2*phi4 - theta*(1-phi1)*phi2
    elif estimator == 'MLATE':
        theta = torch.exp(X@alpha)
        H = Y * theta**(-D)
        f0 = (-(theta+1)*OP+torch.sqrt(OP**2*(theta-1)**2+4*theta*OP)) / (2*theta*(1-OP))
        f1 = f0 * theta
        E = f0*phi1 + (1-phi1)*phi2*phi4/theta + (1-phi1)*(1-phi2)*phi3
    
    if strategy == 'identity':
        return torch.sum((torch.sum(X*((2*Z-1)*(H-E)/f).unsqueeze(1), dim=0))**2)
    elif strategy == 'optimal':
        p011 = (1-phi1)*(1-phi2)*phi3
        p001 = (1-phi1)*(1-phi2)*(1-phi3)
        p110 = (1-phi1)*phi2*phi4
        p100 = (1-phi1)*phi2*(1-phi4)
        p111 = f1*phi1 + p110
        p010 = f0*phi1 + p011
        p101 = 1-p001-p011-p111
        if estimator == 'LATE':
            EH2_1 = p011+p111+ theta**2 * (p111+p101) -2*theta*p111
            EH2_0 = p110+p010+theta**2*(p110+p100) -2*theta*p110
            EH_1 = p111+p011-theta*(p111+p101)
            EH_0 = p110+p010-theta*(p110+p100)
            EZX = (EH2_1-EH_1**2) / d + (EH2_0-EH_0**2)/(1-d)
            w = -X * ((1/torch.cosh(X@alpha)**2) * phi1 / EZX).unsqueeze(1)
            return torch.sum((torch.sum(w*((2*Z-1)*(H-E)/f).unsqueeze(1), dim=0))**2)
        elif estimator == 'MLATE':
            EH2_1 = p111/theta**2+p101
            EH2_0 = p110/theta**2+p100
            EH_1 = p111/theta+p101
            EH_0 = p110/theta+p100
            EZX = (EH2_1-EH_1**2) / d + (EH2_0-EH_0**2)/(1-d)
            w = -X * (1 / theta * f1 * phi1 / EZX).unsqueeze(1)
            return torch.sum((torch.sum(w*((2*Z-1)*(H-E)/f).unsqueeze(1), dim=0))**2)

def MLE(X, Xpl, Xpr, estimator='LATE', dr=True):
    alpha0 = torch.tensor([0.0, -1.0])
    beta0 = (torch.ones(size=(4,2)) * torch.tensor([-0.4,0.8])).T
    eta0 = torch.tensor([-0.4, 1.0])
    gamma0 = torch.tensor([0.1, -1.0])
    Z, D, Y = generate(X, alpha0, beta0, eta0, gamma0, estimator)
    minimum = (-nll(X,Xpl,Xpr,Z,D,Y,alpha0,beta0,eta0,gamma0, estimator)).item()
    # alpha = nn.Parameter(torch.tensor([0.5, 0.5]))
    # beta = nn.Parameter((torch.ones(size=(4,2)) * torch.tensor([-0.5,0.5])).T)
    # eta = nn.Parameter(torch.tensor([-0.5, 0.5]))
    # gamma = nn.Parameter(torch.tensor([0.2, 0.5]))
    alpha = nn.Parameter(torch.rand(size=(2,))*2-1)
    beta = nn.Parameter(torch.rand(size=(2,4))*2-1)
    eta = nn.Parameter(torch.rand(size=(2,))*2-1)
    gamma = nn.Parameter(torch.rand(size=(2,))*2-1)
    opt = torch.optim.Adam(params=(alpha, beta, eta, gamma), lr=1e-3, weight_decay=0)
    optloss = float('inf')
    for i in range(10000):
        opt.zero_grad()
        loss = -nll(X,Xpl,Xpr,Z,D,Y,alpha,beta,eta,gamma, estimator)
        if loss.item() < optloss:
            if abs(loss.item() - optloss) < 1e-6:
                break
            optloss = loss.item()
            mlealpha = alpha.detach().clone()
            mlebeta = beta.detach().clone()
            mleeta = eta.detach().clone()
            mlegamma = gamma.detach().clone()
        # print('Iter {} | loss {:.04f}'.format(i+1, loss.item()))
        loss.backward()
        opt.step()

    if not dr:
        return mlealpha, mlebeta, mleeta, mlegamma, minimum, optloss

    alpha = nn.Parameter(mlealpha.clone(),requires_grad=True)
    # alpha = nn.Parameter(torch.rand(size=(2,))*2-1)
    opt = torch.optim.Adam(params=(alpha,), lr=1e-3, weight_decay=0)
    sqoptloss = float('inf')
    for i in range(10000):
        opt.zero_grad()
        sq_loss = square_loss(X,Xpl,Xpr, Z, D, Y, alpha, mlebeta, mlegamma, mleeta, estimator)
        # print('Iter {} | sq_loss {:.04f}'.format(i+1, sq_loss.item()), alpha)
        if sq_loss.item() < sqoptloss:
            sqoptloss = sq_loss.item()
            drualpha = alpha.detach().clone()
            if abs(sqoptloss) < 1e-6:
                break
        sq_loss.backward()
        opt.step()
    
    alpha = nn.Parameter(mlealpha.clone(),requires_grad=True)
    opt = torch.optim.Adam(params=(alpha,), lr=1e-3, weight_decay=0)
    sqoptloss = float('inf')
    for i in range(10000):
        opt.zero_grad()
        sq_loss = square_loss(X,Xpl,Xpr, Z, D, Y, alpha, mlebeta, mlegamma, mleeta, estimator, strategy='optimal')
        if sq_loss.item() < sqoptloss:
            sqoptloss = sq_loss.item()
            drwalpha = alpha.detach().clone()
            if abs(sqoptloss) < 1e-6:
                break
        sq_loss.backward()
        opt.step()

    return mlealpha, drualpha, drwalpha, mlebeta, mleeta, mlegamma, minimum, optloss


def mle_dr_bad(estimator='LATE'):
    N = 1000
    NR = 1000
    torch.manual_seed(24)
    mlealphas = torch.zeros(size=(NR, 2))
    drualphas = torch.zeros(size=(NR, 2))
    drwalphas = torch.zeros(size=(NR, 2))
    minimums, optlosses = [], []
    X = torch.column_stack((torch.ones(N)*1.0, torch.rand(N)*2-1))
    Xpl = torch.column_stack((torch.ones(N)*1.0, torch.rand(N)*2-1))
    Xpr = torch.column_stack((torch.cat((torch.ones(N//2), torch.zeros(N//2))), torch.cat((torch.zeros(N//2), torch.ones(N//2)))))

    for i in range(NR):
        mlealpha, drualpha,drwalpha, mlebeta, mleeta, mlegamma, minimum, optloss = MLE(X, Xpl, Xpr,  estimator=estimator, dr=True)
        print('{} Experiement | Difference {:.04f} | MLEAlpha: ({:.04f}, {:.04f}) | drualpha: ({:.04f}, {:.04f}) | drwalpha: ({:.04f}, {:.04f})'.format(i+1, optloss-minimum, mlealpha[0].item(), mlealpha[1].item(), drualpha[0].item(), drualpha[1].item(),drwalpha[0].item(), drwalpha[1].item()))
        mlealphas[i] = mlealpha
        drualphas[i] = drualpha
        drwalphas[i] = drwalpha
        minimums.append(minimum)
        optlosses.append(optloss)
    torch.save(mlealphas, 'mle_bad_'+estimator+'.pt')
    torch.save(drualphas, 'dru_bad_'+estimator+'.pt')
    torch.save(drwalphas, 'drw_bad_'+estimator+'.pt')
###
##对初值敏感(没有先验知识时可能造成无法训练出来)，对学习率敏感
##添加贝叶斯先验分布时可能的改进
