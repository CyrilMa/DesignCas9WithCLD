import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import math

import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from scipy.stats import truncnorm

from utils import *

class Layer(nn.Module):
    r"""
    Abstract Class for a Layer of a PGM

    Args:
        name (String): Name of the layer
    """

    def __init__(self, name="layer0"):
        super(Layer, self).__init__()
        self.name = name
        self.full_name = f"Abstract_{name}"
        self.shape = None
        self.gauge = None

    def to(self, device):
        super(Layer, self).to(device)
        return self

    def mean(self, probas):
        return sum(probas)

    def update_params(self):
        return

    def gauge_weights(self):
        return

    def forward(self, *args):
        pass

    def gamma(self, *args):
        pass

    def Z_0(self):
        pass


class DReLULayer(Layer):
    r"""
    Layer of independent dReLU neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """

    def __init__(self, N=100, name="layer0"):
        super(DReLULayer, self).__init__(name)
        self.full_name = f"dReLU_{name}"
        self.N, self.shape = N, N
        self.phi = None
        self.params = nn.ParameterList([nn.Parameter(torch.ones(1,N).float(), requires_grad=True),
                                        nn.Parameter(torch.zeros(1,N).float(), requires_grad=True),
                                        nn.Parameter(torch.zeros(1,N).float(), requires_grad=True),
                                        nn.Parameter(torch.zeros(1,N).float(), requires_grad=True)])
        self.bn = nn.BatchNorm1d(N)

    def update_params(self):
        self.params = nn.ParameterList([nn.Parameter(self.params[0].clamp(0.2, 10000)),
                                        nn.Parameter(self.params[1].clamp(-0.8, 0.8)),
                                        nn.Parameter(self.params[2].clamp(-3, 3)),
                                        nn.Parameter(self.params[3].clamp(-3, 3))])


    def get_slopes(self):
        gamma, mu, theta, delta = self.params
        gamma_plus = gamma/(1+mu)
        gamma_minus = gamma/(1-mu)
        theta_plus = theta + delta/(1+mu)
        theta_minus = theta - delta/(1-mu)
        return gamma_plus, gamma_minus, theta_plus, theta_minus

    def mean(self, probas):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.get_slopes()
        batch_size = probas[0].size(0)
        I = sum([p.view(batch_size, self.N) for p in probas])
        _, _, p_plus, p_minus = self._Z(I)
        mu_plus, mu_minus = (I-theta_plus)/gamma_plus, (I-theta_minus)/gamma_minus
        sigma_plus, sigma_minus = 1/torch.sqrt(gamma_plus), 1/torch.sqrt(gamma_minus)
        p = torch.bernoulli(p_plus)
        mu = p*mu_plus + (1-p)*mu_minus
        mu += 1/math.sqrt(2*math.pi) * (p_plus * torch.exp(-(mu_plus/sigma_plus).pow(2))-p_minus * torch.exp(-(mu_minus/sigma_minus).pow(2)))
        return mu

    def sample(self, probas, beta=1, dilatation = 1):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.get_slopes()
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.N) for p in probas])
        self.phi = phi
        _, _, p_plus, p_minus = self._Z(phi)
        sample_plus = TNP((phi - theta_plus) / gamma_plus, dilatation * torch.sqrt(1 / gamma_plus))
        sample_minus = TNN((phi - theta_minus) / gamma_minus, dilatation * torch.sqrt(1 / gamma_minus))
        return p_plus * sample_plus + p_minus * sample_minus

    def forward(self, h):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.get_slopes()
        h = self.bn(h)
        h_plus = F.relu(h)
        h_minus = -F.relu(-h)
        return (h_plus.pow(2)*gamma_plus/2 + h_minus.pow(2)*gamma_minus/2
                + h_plus*theta_plus + h_minus*theta_minus).sum(-1)

    def gamma(self, iv):
        z_plus, z_minus, _, _ = self._Z(iv)
        return torch.log((z_plus + z_minus).clamp(0.0001, 10000)).sum(-1)

    def _Z(self, x):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.get_slopes()
        r_gamma_plus, r_gamma_minus = torch.sqrt(gamma_plus), torch.sqrt(gamma_minus)
        z_plus = DReLULayer._phi(-(x - theta_plus) / r_gamma_plus) / r_gamma_plus
        z_minus = DReLULayer._phi((x - theta_minus) / r_gamma_minus) / r_gamma_minus
        return z_plus, z_minus, DReLULayer.fillna(z_plus / (z_plus + z_minus)), DReLULayer.fillna(z_minus / (z_plus + z_minus))

    def Z_0(self):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.get_slopes()
        z0 = -0.5 * (torch.log(gamma_plus).sum() + torch.log(gamma_minus).sum())
        z0 = z0 + torch.log(1-phi(theta_plus)).sum() + torch.log(phi(theta_minus)).sum()
        z0 = z0 + self.N * math.log(2*math.pi)
        z0 = z0 + 0.5 * theta_plus.pow(2).sum() + 0.5 * theta_minus.pow(2).sum()
        return z0

    @staticmethod
    def _phi(x):
        r2 = math.sqrt(2)
        rpi2 = math.sqrt(math.pi / 2)
        x = x.clamp(-10,10)
        phix = torch.exp(x ** 2 / 2) * torch.erfc(x / r2) * rpi2
        return phix

    @staticmethod
    def fillna(x, val=1):
        idx = torch.where(x.__ne__(x))[0]
        x[idx] = val
        return x

class GaussianLayer(Layer):
    r"""
    Layer of independent Gaussian neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """

    def __init__(self, N=100, name="layer0"):
        super(GaussianLayer, self).__init__(name)
        self.full_name = f"Gaussian_{name}"
        self.N, self.shape = N, N

    def mean(self, probas):
        batch_size = probas[0].size(0)
        I = sum([p.view(batch_size, self.N) for p in probas])
        return I

    def sample(self, probas, beta=1):
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.N) for p in probas])
        return torch.randn_like(phi)+ phi

    def forward(self, h):
        return -((h).pow(2) / 2).sum(-1)

    def gamma(self, Iv):
        return ((Iv).pow(2) / 2).sum(-1)

    def Z_0(self):
        return self.N / 2 * math.log(2*math.pi)

class OneHotLayer(Layer):
    r"""
    Layer of independent One Hot neurons

    Args:
        weights (torch.FloatTensor): initial weights
        N (Integer): Number of neurons
        q (Integer): Number of values the neuron can take
        name (String): Name of the layer
    """

    def __init__(self, weights=None, N=100, q=21, name="layer0"):
        super(OneHotLayer, self).__init__(name)
        self.full_name = f"OneHot_{name}"
        self.N, self.q, self.shape = N, q, N * q
        self.linear = nn.Linear(self.shape, 1, bias=False)
        self.phi = None
        self.gauge = ZeroSumGauge(self.N, self.q)
        if weights is not None:
            self.linear.weight.data = weights.view(1, -1)

    def to(self, device):
        super(Layer, self).to(device)
        self.gauge = self.gauge.to(device)
        return self

    def gauge_weights(self):
        if self.gauge is not None:
            self.linear.weight = nn.Parameter(self.linear.weight.mm(self.gauge))

    def get_weights(self):
        return self.linear.weight

    def sample(self, probas, beta=1):
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.q, self.N) for p in probas])
        phi += self.linear.weight.view(1, self.q, self.N)
        self.phi = phi
        distribution = OneHotCategorical(probs=F.softmax(phi, 1).permute(0, 2, 1))
        return distribution.sample().permute(0, 2, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.linear(x)

    def l2_reg(self):
        w = self.linear.weight.data
        return w.pow(2).mean()

    def Z_0(self):
        return self.linear.weight.view(self.q, -1).logsumexp(0).sum(0)

EPS = 1e-8
