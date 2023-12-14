import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
from torch import nn

import numpy as np

from utils import *

EPS = 1e-7

def R(W):
    return W.view(*W.size()[:-1], 21, -1).abs().sum(-2)

class AbstractEdge(nn.Module):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out):
        super(AbstractEdge, self).__init__()

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.freeze = True
        self.gauge = None

        self.in_shape, self.out_shape = lay_in.shape, lay_out.shape

    def to(self, device):
        super(AbstractEdge, self).to(device)
        return self

    def freeze(self):
        self.freeze = True

    def unfreeze(self):
        self.freeze = False

    def gauge_weights(self):
        pass

    def get_weights(self):
        pass

    def backward(self, h, sample=True):
        pass

    def forward(self, x, sample=True):
        pass

    def gibbs_step(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        mu = self.linear(x)
        h = mu
        if sample:
            h = self.out_layer.sample([mu])
        else:
            h = self.out_layer.mean([mu])
        mut = self.reverse(h)
        x_rec = mut
        if sample:
            x_rec = self.in_layer.sample([mut])
        else:
            x_rec = self.in_layer.mean([mut])

        return x_rec, h, mut, mu

    def partial_gibbs_step(self, x, active_visible, inactive_units, sample=True):
        m, M = active_visible
        q = self.in_layer.q
        x = x.reshape(x.size(0), -1).float()
        mu = h = self.forward(x, sample=False)
        if sample:
            h = self.out_layer.sample([mu])
        else:
            h = self.out_layer.mean([mu])
        h[:,inactive_units] = mu[:,inactive_units]
        mut = x_rec = self.backward(h, sample=False)
        if sample:
            x_rec = self.in_layer.sample([mut])
        else:
            x_rec = self.in_layer.mean([mut])
        x_rec, x, mut = x_rec.view(x.size(0),q,-1), x.view(x.size(0),q,-1), mut.view(mut.size(0),q,-1)
        x_rec[:,:,:m] = x[:,:,:m]; x_rec[:,:,M:] = x[:,:,M:]
        mut[:,:,:m] = x[:,:,:m]; mut[:,:,M:] = x[:,:,M:]
        return x_rec.reshape(x.size(0), -1), h, mut.reshape(x.size(0), -1).detach(), mu.detach()

    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        pass

    def l1c_reg(self):
        pass

    def save(self, filename):
        torch.save(self, filename)


class Edge(AbstractEdge):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out, gauge=None, weights=None):
        super(Edge, self).__init__(lay_in, lay_out)

        # Gauge
        self.gauge = None
        if gauge is not None:
            self.gauge = gauge.detach()
            self.gauge = self.gauge.to(device)

        # Model
        self.linear = nn.Linear(self.in_shape, self.out_shape, bias = False)
        nn.init.xavier_uniform(self.linear.weight)
        self.reverse = nn.Linear(self.out_shape, self.in_shape, bias = False)
        if weights is not None:
            self.linear.weight = weights
        self.reverse.weight.data = self.linear.weight.data.t()

    def to(self, device):
        super(Edge, self).to(device)
        self.gauge = self.gauge.to(device)
        return self

    def sim_weights(self, strength = 1):
        W = self.linear.weight
        N = self.out_shape
        return sum(sum(torch.exp(-(W[i]*W[j]).mean(-1) for j in range(i)) for i in range(N)))/(N*(N-1)/2)

    def gauge_weights(self):
        if self.gauge is not None:
            self.linear.weight.data = nn.Parameter(self.linear.weight.data.mm(self.gauge))
        self.reverse.weight.data = self.linear.weight.data.t()

    def get_weights(self):
        if self.gauge is not None:
            return self.linear.weight.mm(self.gauge)
        return self.linear.weight

    def backward(self, h, sample=True):
        h = h.reshape(h.size(0), -1)
        p = self.reverse(h)
        if sample:
            x_rec = self.in_layer.sample([p])
        else:
            x_rec = self.in_layer.mean([p])
        return x_rec

    def forward(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        p = self.linear(x)
        if sample:
            h = self.out_layer.sample([p])
        else:
            h = self.out_layer.mean([p])
        return h

    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.abs(w).sum(-1)
        return reg.pow(2).sum(0)

    def l1c_reg(self):
        W = self.get_weights()
        rw = R(W)
        rw = (rw[:,None]*rw[:,:,None])
        rw = torch.stack([torch.diagonal(rw,k,1,2).sum(-1) for k in range(1,rw.size(-1))],-1)
        reg = (rw * torch.tensor([1-np.exp(-i) for i in range(1,rw.size(-1)+1)])[None]).mean(-1)
        return reg.sum()

class InvertibleLinear(nn.Module):
    def __init__(self, in_features, out_features,):
        super(InvertibleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(2*(torch.rand(self.out_features, self.in_features)-0.5)/math.sqrt(self.in_features))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).permute(1,0)
        h = self.weight[None].matmul(x).permute(2,0,1)
        return h

    def backward(self, h):
        batch_size = h.size(0)
        h = h.view(batch_size, -1, 1) * self.weight
        x = self.weight.t()[None].matmul(h)
        return x

class FilterbankLinear(nn.Module):
    def __init__(self, in_features, in_channels, window_size, strides, out_channels):
        super(FilterbankLinear, self).__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.window_size = window_size
        self.strides = strides
        self.out_channels = out_channels
        self.Nk, self.fbank = self.build_fbank()
        self.weight = nn.Parameter(2*(torch.rand(self.Nk, self.window_size*self.in_channels)-0.5)/math.sqrt(self.window_size*self.in_channels))

    def to(self, device):
        super(FilterbankLinear, self).to(device)
        self.fbank = self.fbank.to(device)
        return self

    def build_fbank(self):
        Nk = self.Nk = self.out_channels*(int((self.in_features-self.window_size)/self.strides - EPS)+1)
        filters = torch.zeros(self.Nk, self.in_channels, self.window_size, self.in_channels, self.in_features)
        cursor = 0
        for j in range(int((self.in_features-self.window_size)/self.strides - EPS)):
            for k in range(self.in_channels):
                filters[self.out_channels*j:self.out_channels*(j+1),k, torch.arange(self.window_size), k, cursor+torch.arange(self.window_size)]=1
            cursor += self.strides
        for k in range(self.in_channels):
            filters[self.out_channels*int((self.in_features-self.window_size)/self.strides - EPS):, k, torch.arange(self.window_size),k,self.in_features-self.window_size+torch.arange(self.window_size)] = 1

        filters = filters.view(self.Nk, self.in_channels * self.window_size, self.in_channels * self.in_features)
        return Nk, to_sparse(filters)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).t()
        fx = torch.sparse.mm(self.fbank,x).t().view(batch_size, self.Nk, -1)
        h = fx * self.weight[None]
        return h.sum(-1)

    def backward(self, h):
        batch_size = h.size(0)
        wh = (h.view(batch_size, -1, 1) * self.weight[None]).view(batch_size,-1).t()
        x = torch.sparse.mm(self.fbank.t(),wh).t()
        return x


class FilterbankEdge(AbstractEdge):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out, window_size, strides, out_channels, weights=None):
        super(FilterbankEdge, self).__init__(lay_in, lay_out)

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.N, self.q = lay_in.N, lay_in.q
        self.window_size, self.strides, self.out_channels = window_size, strides, out_channels

        self.Nks = FilterbankEdge._Nks(self.N, window_size, strides, out_channels)
        self.cumNks = [0]+list(np.cumsum(self.Nks))
        self.gauges = [ZeroSumGauge(w, self.q).to(device) for w in window_size]

        # Model
        self.linears = nn.ModuleList([FilterbankLinear(self.N, self.q, w, s, d) for w, s, d in zip(window_size, strides, out_channels)])
        for linear in self.linears:
            nn.init.xavier_uniform(linear.weight)
        if weights is not None:
            self.linear.weight = weights

    def to(self, device):
        super(FilterbankEdge, self).to(device)
        for lin in self.linears:
            lin.to(device)
        self.gauges = [gauge.to(device) for gauge in self.gauges]
        return self

    def gauge_weights(self):
        for linear, gauge in zip(self.linears, self.gauges):
            linear.weight.data = nn.Parameter(linear.weight.data.mm(gauge))

    def get_weights(self):
        if self.gauges is not None:
            return [linear.weight.mm(gauge) for linear, gauge in zip(self.linears, self.gauges)]
        return [linear.weight for linear in self.linears]

    def backward(self, h, sample=True):
        h = h.reshape(h.size(0), -1)
        p = torch.stack([linear.backward(h[:,m:M]) for linear, m, M in zip(self.linears, self.cumNks[:-1], self.cumNks[1:])],-1).sum(-1)
        if sample:
            x = self.in_layer.sample([p])
            return x
        return p

    def forward(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        p = torch.cat([linear(x) for linear in self.linears],-1)
        if sample:
            h = self.out_layer.sample([p])
            return h
        return p

    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([math.sqrt(self.N/size) * torch.abs(w_).sum(-1) for size, w_ in zip(self.window_size, w)],-1)
        return reg.pow(2).sum(0)

    def linfb_reg(self):
        w = self.get_weights()
        reg = torch.cat([torch.max(w_)[0].sum(-1) for size, w_ in zip(self.window_size, w)],-1)
        return reg.pow(2).sum(0)


    def l2_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.pow(2).sum(-1) for w_ in w],-1)
        return reg.pow(2).sum(0)

    def _Nks(N,window_size,strides,out_channels):
        return [(d*(int((N-w)/s - EPS)+1)) for w,s,d in zip(window_size,strides,out_channels)]

class ConvolutionalLinear(nn.Module):
    def __init__(self, in_features, in_channels, window_size, strides, out_channels, opad = 0):
        super(ConvolutionalLinear, self).__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.window_size = window_size
        self.strides = strides
        self.out_channels = out_channels
        weights = torch.rand
        conv = nn.Conv1d(in_channels, out_channels, window_size, stride = strides, bias = False)
        self.weight = nn.Parameter(torch.randn_like(conv.weight.data))
        self.conv = (lambda x : F.conv1d(x, self.weight, stride = strides))
        self.reverse = (lambda h : F.conv_transpose1d(h, self.weight, stride = strides, output_padding = opad))

    def to(self, device):
        super(ConvolutionalLinear, self).to(device)
        return self

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_channels, -1)
        h = self.conv(x)
        return h.reshape(batch_size,-1)

    def backward(self, h):
        batch_size = h.size(0)
        h = h.reshape(batch_size,self.out_channels, -1)
        x = self.reverse(h)
        return x.view(batch_size,-1)


class ConvolutionalEdge(AbstractEdge):
    r"""
    Class to handle the original Edge of the RBM

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies)
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (Optional(torch.FloatTensor)): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """

    def __init__(self, lay_in, lay_out, window_size, strides, out_channels, weights=None):
        super(ConvolutionalEdge, self).__init__(lay_in, lay_out)

        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.N, self.q = lay_in.N, lay_in.q

        self.Nks = ConvolutionalEdge._Nks(self.N, window_size, strides, out_channels)
        self.opads = [self.N - ((Nk//d-1)*s+w) for Nk,s,w,d in zip(self.Nks, strides, window_size,out_channels)]
        self.cumNks = [0]+list(np.cumsum(self.Nks))
        self.gauges = [ZeroSumGauge(w, self.q).to(device) for w in window_size]

        # Model
        self.linears = nn.ModuleList([ConvolutionalLinear(self.N, self.q, w, s, d, opad) for w, s, d, opad in zip(window_size, strides, out_channels, self.opads)])
        for linear in self.linears:
            nn.init.xavier_uniform(linear.weight)
        if weights is not None:
            self.linear.weight = weights

    def to(self, device):
        super(ConvolutionalEdge, self).to(device)
        for lin in self.linears:
            lin.to(device)
        self.gauges = [gauge.to(device) for gauge in self.gauges]
        return self

    def gauge_weights(self):
        for linear, gauge in zip(self.linears, self.gauges):
            size = linear.weight.data.size()
            linear.weight.data = nn.Parameter(linear.weight.data.reshape(linear.out_channels,-1).mm(gauge).reshape(*size))

    def get_weights(self):
        if self.gauges is not None:
            return [linear.weight.reshape(linear.out_channels,-1).mm(gauge) for linear, gauge in zip(self.linears, self.gauges)]
        return [linear.weight for linear in self.linears]

    def backward(self, h, sample=True):
        h = h.reshape(h.size(0), -1)
        p = torch.stack([linear.backward(h[:,m:M]) for linear, m, M in zip(self.linears, self.cumNks[:-1], self.cumNks[1:])],-1).sum(-1)
        if sample:
            x = self.in_layer.sample([p])
            return x
        return p

    def forward(self, x, sample=True):
        x = x.reshape(x.size(0), -1)
        p = torch.cat([linear(x) for linear in self.linears],-1)
        if sample:
            h = self.out_layer.sample([p])
            return h
        return p

    def l1b_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.abs(w_).sum(-1) for w_ in w],-1)
        return reg.pow(2).sum(0)

    def l2_reg(self):
        r"""
        Evaluate the L1b factor for the weights of an edge
        """
        w = self.get_weights()
        reg = torch.cat([torch.pow(2).sum(-1) for w_ in w],-1)
        return reg.pow(2).sum(0)

    def _Nks(N,window_size,strides,out_channels):
        return [d*int((N-(w-1)-1)/s+1) for w,s,d in zip(window_size,strides,out_channels)]
