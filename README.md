# Introduction
Cofounding variable的存在可能造成因果效应的误判. 而使用instrumental variable可以解决这一问题. 

Def: An instrumental variable is a pre-treatment variable covariate that associate with the outcome only through its effect on the treatment.

前人工作中, ATE的估计主要有两种方案:
- 给予homogeneity assumptions involving unmeasured confounders. (untestable).
- estimate local ATE, under a certain monotonicity assumption.

将local ATE估计方法用于binary outcome的主要问题是: 大多数都基于additive local ATE的模型, 而不是更适用于binary outcome的multiplicative的模型.

本文提出了estimating procedures for both the additive and the multiplicative local ATE with a binary outcome.

# Framework, Notation and Existing Estimators
## Notations
- Binary exposure indicator $D$. (每个个体接受的处理). $D(z)$ defines the potential exposure if $Z=z$($D(z)$ is a random variable).
- Binary outcome $Y$. $Y(z,d)$ is the outcome that would have been observed (also random variable).
- Two confounding variables: observed $X$, unobserved $D$. 
- Observed binary instrumental variable $Z$.

target: estimating the conditional treatment effects in the complier stratum:
$$LATE(X) = E(Y(1)-Y(0)|D(1)>D(0), X)$$
$$MLATE(X) = E(Y(1)|D(1)>D(0), X) / E(Y(0)|D(1)>D(0), X)$$

target: estimating local ATE $\delta^L(X)$ and $\delta^M(X)$. 

method:
- Abadie(2003). 用logistic model to parametrize the **local average treatment effect** $E(Y(d)|D(1)>D(0), X)$. 问题: 模型可解释性差. 需要明确$p(Z=1|X)$.
- Okui(2012), Ogburn(2015). 

# Simulation
We generate data following the same setup in the paper. Our model shares **the same functional form** with the data generating models. 

Three parts of models:
- model of interest (target): $\theta(X) = \delta^M(X) or \delta^L(X)$. Our parameterization is $\theta(X,\alpha)$.
- instrumental density model. $p(Z|X,\gamma)$.
- Other nuisance model (5 totally). $\phi_i(X,\beta), i=1,2,3,4$, $OP^{CO}(X,\eta)$.

Theorem 1 says there exists one bijection between likelihood and our models. Thus we can
- naively use MLE to estimate the parameters.
- firstly obtain MLE estimator, then use estimating equation with identity weight to calculate the $\hat{\alpha}_{dr}$.
- firstly obtain MLE estimator, then use estimating equation with optimal weight to calculate the $\hat{\alpha}_{dr}$.

Models are correctly specified or not: $X$ (true), $X^+$ (misspecified for instrumental density model), $X'$ (misspecified for other nuisance model). The model $\theta(X)$ is always correctly specified.
- bth: $X$ is used in all nuisance models;
- psc: $X$ is used in the instrumental density model, and $X'$ is used in other nuisance models;
- opc: $X^+$ is used in the instrumental density model, and $X$ is used in other nuisance models;
- bad: $X^+$ is used in the instrumental density model, and $X'$ is used in other nuisance models.

About results analysis.
- $Bias = |E(\hat{\alpha})-\alpha|$. S.E. = sample std. We can estimate them by drawing $Z, D, Y$ samples, and repeating this experiments for 1000 times. 
- For each time, we can calculate empirical C.I. using 500 bootstrap samples. Therefore, we can obtain 1000 confidence intervals. Then the converage probability and average C.I. length can be calculated. 

(We will optimize 500000 models!!)

# Real Data
- target: whether 401(k) contributions represent additional savings or simply replace other retirement plans.
- Unobserved cofounders: the underlying preference for savings. Eligibility is determined by employers individual prederences.
- 