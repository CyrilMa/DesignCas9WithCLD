import sys
import os
import random
import time

sys.path.append(os.path.dirname(os.getcwd()))

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn.functional as F

from utils import *


def classifier_from_x(classifier, edge, target):
    """ Create a classifier from a model and an edge.

    Args:
        classifier (nn.Module): Classifier Model.
        edge (nn.Module): Edge.
        target (torch.Tensor): Target tensor.

    Returns:
        crit (function): Classifier.
    """
    target = target

    def crit(x):
        p = (classifier(edge(x, False))).sigmoid()
        return (target * (p + 1e-7).log() + (1 - target) * (1 - p + 1e-7).log()).mean(-1)

    return crit


class Walker(object):
    """ Walker class for CLD.
    
    Args:
        x0 (torch.Tensor): Initial input.
        model (nn.Module): Model.
        objective (function): Objective function.
        constraints (list): List of constraints. 
        zero_idx (list): List of indices to zero.
        gamma (float): Gamma.
        n (int): Number of samples to try at each step.
        a (float): size of the step.
        c (float): norm regularization factor.
        eps (float): standard deviation of the standard scaler.
        T (float): Sampling temperature.
        target (torch.Tensor): Target tensor.
        weight_constraints (list): List of weights for constraints.
        kept_idx (list): List of indices to keep.
        n_samples (int): Number of samples.

    Attributes:
        objective (function): Objective function.
        constraints (list): List of constraints.
        weight_constraints (list): List of weights for constraints.
        n_samples (int): Number of samples.
        target (torch.Tensor): Target tensor.
        zero_idx (list): List of indices to zero.
        a (float): size of the step.
        c (float): norm regularization factor.
        eps (float): standard deviation of the standard scaler.
        n (int): Number of samples to try at each step.
        gamma (float): Gamma.
        model (nn.Module): Model.
        pi (nn.Module): Pi layer.
        edge (nn.Module): Edge.
        Nh (int): Number of hidden units.
        q (int): Number of input channels.
        N (int): Number of input units.
        Z (torch.Tensor): Z tensor (partition function of the model).
        classifier (function): Classifier.
        device (torch.device): Device.
        kept_idx (list): List of indices to keep.
        x0 (torch.Tensor): Initial input.
        h0 (torch.Tensor): Initial hidden state.
        e0 (torch.Tensor): Initial znergy (of the initial input).
        p0 (torch.Tensor): Initial classifier probability (of the initial input).
        TRACKS (list): List of tracks.
        track (list): List of tracks.
        writer (SummaryWriter): Tensorboard writer.
    """

    def __init__(self, x0, model, objective, constraints, zero_idx, gamma, n, a, c, eps, T,
                 target=None, weight_constraints=None, kept_idx=None, n_samples=16):
        super(Walker, self).__init__()
        self.track = None
        self.writer = None
        self.objective = objective
        self.constraints = constraints
        self.n_samples = n_samples
        self.target = target
        self.zero_idx = zero_idx

        self.a = a
        self.c = c
        self.eps = eps
        self.n = n
        self.gamma = gamma
        self.T = T

        self.model = model
        self.pi = model.layers["pi"]
        self.edge = model.edges["pi -> hidden"]
        self.Nh = model.layers["hidden"].N
        self.q = self.pi.q
        self.N = self.pi.N
        self.Z = model.Z
        self.classifier = classifier_from_x(model.classifier, self.edge, self.target)
        self.device = model.device

        if kept_idx is None:
            kept_idx = range(self.N)
        self.kept_idx = kept_idx

        self.x0 = x0.reshape(1, self.q, self.N).float()
        self.h0 = self.edge(self.x0.view(1, self.q, -1)[:, :, ], False).detach()
        self.e0 = model(
            {"pi": self.x0.view(1, self.q, -1)[:, :, self.kept_idx].to(self.device)}).detach().cpu() / self.N - self.Z
        self.p0 = self.classifier(self.x0.view(1, self.q, -1)[:, :, self.kept_idx]).exp().detach()

        self.TRACKS = []

        self.weight_constraints = [10 for _ in constraints]
        if weight_constraints is not None:
            self.weight_constraints = weight_constraints

    def write_tensorboard(self, logs, n_iter):
        """ Write logs to tensorboard.

        Args:
            logs (dict): Dictionary of logs.
            n_iter (int): Number of iterations.
        """

        for k, v in logs.items():
            self.writer.add_scalar(k, v, n_iter)

    def run(self, batch_size=32, n_epochs=500, verbose=True):
        """ Run the walker.

        Args:
            batch_size (int): Batch size.
            n_epochs (int): Number of epochs.
            verbose (bool): Verbose.
        """
        self.writer = SummaryWriter(f"{DATA}/tensorboard/walker/test_{int(time.time())}")
        self.track = dict()
        e_ = torch.cat([self.e0.clone() for _ in range(batch_size)], 0)
        h_ = torch.cat([self.h0.clone() for _ in range(batch_size)], 0)
        x_ = torch.cat([self.x0.reshape(1, -1).clone() for _ in range(batch_size)], 0)
        p_ = torch.cat([self.p0.clone() for _ in range(batch_size)], 0)
        diff_ = p_.clone()
        dynamic_ = torch.ones(batch_size)
        abs_diff_ = torch.zeros(batch_size)
        n_mut = 0
        for i in range(1, n_epochs):
            h, x, gradient_C, gradient_phi = self.step(x_, h_)
            with torch.no_grad():
                x = x.float().cpu().reshape(batch_size * self.n, -1)
                h = h.float().cpu().reshape(batch_size * self.n, -1)

                abs_diff = (x.reshape(batch_size * self.n, self.q, -1).argmax(1) != self.x0.argmax(1)).sum(-1)
                e = (self.model({"pi": x.reshape(batch_size * self.n, self.q, -1)[:, :, self.kept_idx]}) / self.N -
                     self.Z)
                p = self.classifier(x.reshape(batch_size * self.n, self.q, -1)[:, :, self.kept_idx]).exp()

                diff = p
                diff = torch.cat([diff.reshape(-1, self.n), diff_[:, None]], -1)
                diff[:, -1] = -1e5
                max_idx = diff.argmax(1).long()
                diff = diff.reshape(-1)
                dynamic_ = (max_idx < self.n).int() + 1 * dynamic_
                changed_idx = torch.where(max_idx < self.n)[0]
                killed_idx = torch.where(torch.rand(batch_size) > dynamic_)[0]
                probs = dynamic_.cumsum(0) / dynamic_.sum()
                replaced_idx = torch.tensor([(probs < random.random()).sum().int().item() for _ in killed_idx])
                if len(changed_idx) > 0:
                    for j in changed_idx:
                        x_[j] = x[self.n * j + max_idx[j]]
                        h_[j] = h[self.n * j + max_idx[j]]
                        e_[j] = e[self.n * j + max_idx[j]]
                        p_[j] = p[self.n * j + max_idx[j]]
                        abs_diff_[j] = abs_diff[self.n * j + max_idx[j]]
                    diff_[changed_idx] = diff[(self.n + 1) * changed_idx + max_idx[changed_idx]]
                if len(killed_idx) > 0:
                    x_[killed_idx] = x_[replaced_idx]
                    h_[killed_idx] = h_[replaced_idx]
                    e_[killed_idx] = e_[replaced_idx]
                    p_[killed_idx] = p_[replaced_idx]
                    abs_diff_[killed_idx] = abs_diff_[replaced_idx]
                    diff_[killed_idx] = diff_[replaced_idx]
                    dynamic_[killed_idx] = dynamic_[replaced_idx]

                n_mut += len(changed_idx)
                logs = {"e": e_.clone().numpy(), "abs_diff": abs_diff_.clone().numpy(),
                        "dynamic": dynamic_.clone().numpy(), "diff": diff_.clone().numpy(),
                        "h": h_.clone().pow(2).sum(-1).sqrt().numpy()}
                self.TRACKS.append(logs)
                if verbose:
                    print(f"""{n_mut}/{batch_size * i} [{(100 * n_mut) / (batch_size * i):.2f}%] 
                    || E = {e_.mean():.3f} 
                    || C = {p_.mean():.3f} 
                    || Dynamic : {list(e_.detach().numpy())}
                    || Abs Diff : {abs_diff_.mean().detach().item():.3f}
                    || h = {h_.pow(2).sum(-1).sqrt().mean():.3f}""")
        return x_

    def step(self, xt, ht, ):
        """ Step of the walker.

        Args:
            xt (torch.Tensor): Current state.
            ht (torch.Tensor): Current hidden state.

        Returns:
            ht (torch.Tensor): Next hidden state.
            xt (torch.Tensor): Next state.
            gradient_C (torch.Tensor): Gradient of the classifier.
            constraint_gradient_phi (torch.Tensor): Gradient of the constraint.
        """
        batch_size = xt.size(0)
        samples = self.n_samples
        mut = self.edge.reverse(ht.repeat(samples, 1).reshape(samples * batch_size, -1))
        mut = mut.reshape(mut.size(0), self.q, -1)
        mut[:, 0] = -10000
        mut[:, 0, self.zero_idx] = 10000

        probs = (mut + self.pi.linear.weight.view(1, self.q, self.N)) / self.T
        distribution = OneHotCategorical(probs=F.softmax(probs, 1).permute(0, 2, 1))
        x_ = distribution.sample().permute(0, 2, 1)
        x = self.x0.clone().repeat(samples * batch_size, 1, 1).reshape(samples * batch_size, self.q, -1)
        x[:, :] = x_.reshape(samples * batch_size, self.q, -1)
        if self.objective is None:
            hs = [torch.tensor(h, requires_grad=True) for h in ht]
            p = [(1 * self.classifier(h_[None]))[0].sigmoid() for h_ in hs]
            crits = [((self.target * (p_ + 1e-7).log() + (1 - self.target) * (1 - p_ + 1e-7).log()).mean(-1)).mean(0)
                     for p_ in p]
            [crit_.backward() for crit_ in crits]
            gradient_C = self.a * (
                    torch.stack([h_.grad for h_ in hs], 0) + self.eps * torch.randn_like(ht) - self.c * ht)

        with torch.no_grad():
            Wv = self.edge(x_, False).reshape(samples, batch_size, -1)
            x = x.reshape(samples, batch_size, -1)
            if self.objective is not None:
                objective_phi, objective_gradient_phi = self.objective.run(x, Wv)
                gradient_C = self.a * (objective_gradient_phi + self.eps * torch.randn_like(ht) - self.c * ht)

            constraint_phi = torch.zeros(batch_size, )
            constraint_gradient_phi = torch.zeros(batch_size, self.Nh)
            for constraint, w in zip(self.constraints, self.weight_constraints):
                phi_, gradient_phi_ = constraint.run(x, Wv)
                constraint_phi += w * phi_
                constraint_gradient_phi += w * gradient_phi_

        norm_phi2 = constraint_gradient_phi.pow(2).sum(-1).detach()
        angle_C = (constraint_gradient_phi * gradient_C).sum(-1)
        diff = constraint_phi / self.gamma
        bt = diff + angle_C
        idx = torch.where(norm_phi2 > 0)
        bt[idx] = (bt[idx] / norm_phi2[idx])
        bt, gradient_C, constraint_gradient_phi = bt[None], gradient_C[None], constraint_gradient_phi[None]
        h = ht.repeat(self.n, 1).reshape(self.n, batch_size, -1)
        h = (h + self.gamma * (gradient_C - bt[:, :, None] * constraint_gradient_phi).clip(-10, 10)).view(batch_size *
                                                                                                          self.n, -1)

        mut = self.edge.reverse(h).reshape(-1, self.q, self.N)
        mut[:, 0] = -10000
        mut[:, 0, self.zero_idx] = 10000
        probs = (mut + self.pi.linear.weight.view(1, self.q, self.N)) / self.T
        distribution = OneHotCategorical(probs=F.softmax(probs, 1).permute(0, 2, 1))
        x_ = distribution.sample().permute(0, 2, 1)
        x = self.x0.clone().repeat(self.n * batch_size, 1, 1).reshape(self.n * batch_size, self.q, -1)
        x[:, :, self.kept_idx] = x_.reshape(self.n * batch_size, self.q, -1)
        h = h.reshape(self.n, batch_size, -1).permute(1, 0, 2)
        x = x.reshape(self.n, batch_size, self.q, -1).permute(1, 0, 2, 3)

        return h, x, gradient_C, constraint_gradient_phi
