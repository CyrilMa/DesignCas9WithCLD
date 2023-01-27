import sys, os
import multiprocessing
import torch
from copy import deepcopy

sys.path.append(os.path.dirname(os.getcwd()))
from utils import *

device = "cpu"


class AbstractCriterion(object):
    def __init__(self, name):
        super(AbstractCriterion, self).__init__()
        self.name = name

    def run(self, x, Wv):
        pass

    def gradient_phi(self, x):
        pass


class FoldXCriterion(AbstractCriterion):
    def __init__(self, structure, postprocessing=None, chain="A", position=0, name="fx"):
        super(FoldXCriterion, self).__init__(name)
        self.structure = structure
        self.chain = chain
        self.position = position
        self.postprocessing = (lambda x: x)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        samples, batch_size = x.size(0), x.size(1)
        fx = self.apply(x.reshape(samples * batch_size, 21, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(fx)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        with multiprocessing.Pool(processes=32) as pool:
            seqs = []
            for x_ in x:
                seqs.append([AA[i] for i in x_[1:, nnz_idx].cpu().argmax(0)])

            energies = pool.map(FoldXCriterion.aux,
                                [(x_, deepcopy(self.structure), self.position, self.chain) for x_ in seqs])
        return torch.tensor(energies)

    def aux(args):
        seq, s, position, chain = args

        for x in seq:
            if position in s.data[chain].keys():
                res = s.data[chain][position]
                res.code = pl1to3[x]
                s.data[chain][position] = res
            position += 1
        return float(s.getTotalEnergy().loc["model"]["total"])


class InterpoleCriterion(AbstractCriterion):
    def __init__(self, x0, x1, idx, postprocessing=None, N=500, name="interpole"):
        super(InterpoleCriterion, self).__init__(name)
        self.x0 = x0.reshape(1, 21, -1)[:, :, idx]
        self.x1 = x1.reshape(1, 21, -1)[:, :, idx]
        self.idx = idx
        self.N = N
        self.postprocessing = (lambda x: x)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv, t):
        samples, batch_size = x.size(0), x.size(1)
        sim = self.apply(x.reshape(samples * batch_size, -1), t).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(sim)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x: torch.Tensor, t):
        x0 = self.x0.reshape(1, 21, -1)
        x1 = self.x1.reshape(1, 21, -1)

        x = x.reshape(x.size(0), 21, -1)[:, :, self.idx]
        dist_x = (x[:, :, :].argmax(1) != x0[:, :, :].argmax(1)).int().float().sum(-1)
        dist_y = (x[:, :, :].argmax(1) != x1[:, :, :].argmax(1)).int().float().sum(-1)
        return (dist_x / (dist_x + dist_y) - t / self.N).pow(2)


class SimCriterion(AbstractCriterion):
    def __init__(self, x0, nnz_idx, postprocessing=None, name="sim"):
        super(SimCriterion, self).__init__(name)
        self.postprocessing = (lambda x: x)
        self.nnz_idx = nnz_idx
        self.x0 = x0.reshape(1, 21, -1)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        samples, batch_size = x.size(0), x.size(1)
        sim = self.apply(x.reshape(samples * batch_size, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(sim)

        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        x0 = self.x0.reshape(1, 21, -1)
        x = x.reshape(x.size(0), 21, -1)
        return (x[:, :, self.nnz_idx].argmax(1) != x0[:, :, self.nnz_idx].argmax(1)).int().float().sum(-1)


class RbmCriterion(AbstractCriterion):
    def __init__(self, model, postprocessing=None, name="erbm"):
        super(RbmCriterion, self).__init__(name)
        self.postprocessing = (lambda x: x)
        self.model = model
        self.Z = self.model.Z
        self.N = self.model.layers["pi"].N
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        samples, batch_size = x.size(0), x.size(1)
        e = self.apply(x.reshape(samples, batch_size, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(e)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        samples, batch_size = x.size(0), x.size(1)
        return self.model({"pi": x.reshape(samples * batch_size, 21, N)[:, :, :]}) / 736 - self.Z


class MultiFoldXCriterion(AbstractCriterion):
    def __init__(self, structures, postprocessing=None, chains=None, positions=None, name="fx"):
        super(MultiFoldXCriterion, self).__init__(name)
        self.structures = structures
        self.chains = chains
        self.positions = positions
        self.postprocessing = (lambda x: x)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        samples, batch_size = x.size(0), x.size(1)
        fx1 = self.apply1(x.reshape(samples * batch_size, 21, -1)).to(device).reshape(samples, batch_size, -1)
        fx2 = self.apply2(x.reshape(samples * batch_size, 21, -1)).to(device).reshape(samples, batch_size, -1)
        phi = fx1 - fx2
        phi_ = phi.mean(0)
        Wv_ = Wv.mean(0)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply1(self, x):
        structure, position, chain = self.structures[0], self.positions[0], self.chains[0]
        with multiprocessing.Pool(processes=32) as pool:
            seqs = []
            for x_ in x:
                seqs.append([AA[i] for i in x_[1:, nnz_idx].cpu().argmax(0)])

            energies = pool.map(MultiFoldXCriterion.aux, [(x_, deepcopy(structure), position, chain) for x_ in seqs])
        return torch.tensor(energies)

    def apply2(self, x):
        structure, position, chain = self.structures[1], self.positions[1], self.chains[1]
        with multiprocessing.Pool(processes=32) as pool:
            seqs = []
            for x_ in x:
                seqs.append([AA[i] for i in x_[1:, nnz_idx].cpu().argmax(0)])

            energies = pool.map(MultiFoldXCriterion.aux, [(x_, deepcopy(structure), position, chain) for x_ in seqs])
        return torch.tensor(energies)

    def aux(args):
        seq, s, position, chain = args

        for x in seq:
            if position in s.data[chain].keys():
                res = s.data[chain][position]
                res.code = pl1to3[x]
                s.data[chain][position] = res
            position += 1
        return float(s.getTotalEnergy().loc["model"]["total"])
