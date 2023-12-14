import sys, os
import multiprocessing
import torch
from copy import deepcopy
from pyfoldx.structure import Structure
sys.path.append(os.path.dirname(os.getcwd()))
from utils import *

device = "cpu"

class AbstractCriterion(object):
    """ Abstract class for a criterion."""

    def __init__(self, name):
        super(AbstractCriterion, self).__init__()
        self.name = name

    def run(self, x, Wv):
        pass

    def gradient_phi(self, x):
        pass


class FoldXCriterion(AbstractCriterion):
    """ FoldX criterion.

    Args:
        structure (pyfoldx.structure.Structure): Structure file.
        chain (str): The chain to consider in a FoldX file.
        position (int): Position.
        postprocessing (AbstractPostprocessor): Postprocessing function.
        name (str): Name of the criterion to display.
    """

    def __init__(self, structure, postprocessing=None, chain="A", position=0, name="fx"):
        super(FoldXCriterion, self).__init__(name)
        self.structure = structure
        self.chain = chain
        self.position = position
        self.postprocessing = (lambda x: x)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        """ Run the criterion.

        Args:
            x (torch.Tensor): Input tensor.
            Wv (torch.Tensor): Weight tensor.

        Returns:
            phi (torch.Tensor): Criterion value.
            gradient_phi (torch.Tensor): Criterion gradient.
        """
        samples, batch_size = x.size(0), x.size(1)
        fx = self.apply(x.reshape(samples * batch_size, 21, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(fx)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        """ Apply the criterion.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: FoldX energy.
        """
        with multiprocessing.Pool(processes=32) as pool:
            seqs = []
            for x_ in x:
                seqs.append([AA[i] for i in x_[1:, nnz_idx].cpu().argmax(0)])

            energies = pool.map(FoldXCriterion.aux,
                                [(x_, deepcopy(self.structure), self.position, self.chain) for x_ in seqs])
        return torch.tensor(energies)

    def aux(args):
        """ Auxiliar function for multiprocessing.

        Args:
            args: Arguments.

        Returns:
            float: FoldX energy.
        """
        seq, s, position, chain = args

        for x in seq:
            if position in s.data[chain].keys():
                res = s.data[chain][position]
                res.code = pl1to3[x]
                s.data[chain][position] = res
            position += 1
        return float(s.getTotalEnergy().loc["model"]["total"])


class InterpoleCriterion(AbstractCriterion):
    """ Interpole criterion.

    Args:
        x0 (torch.Tensor): Input tensor source.
        x1 (torch.Tensor): Input tensor target.
        idx (int): Index of the amino acid to interpolate.
        postprocessing (function): Postprocessing function.
        N (int): Number of interpolations.
        name (str): Name of the criterion.
    """

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
        """ Run the criterion.

        Args:
            x (torch.Tensor): Input tensor.
            Wv (torch.Tensor): Weight tensor.
            t (float): Interpolation parameter.

        Returns:
            phi (torch.Tensor): Criterion value.
            gradient_phi (torch.Tensor): Criterion gradient.
        """
        samples, batch_size = x.size(0), x.size(1)
        sim = self.apply(x.reshape(samples * batch_size, -1), t).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(sim)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x: torch.Tensor, t):
        """ Apply the criterion.

        Args:
            x (torch.Tensor): Input tensor.
            t (float): Interpolation parameter.

        Returns:
            torch.Tensor: Interpolation criterion.
        """
        x0 = self.x0.reshape(1, 21, -1)
        x1 = self.x1.reshape(1, 21, -1)

        x = x.reshape(x.size(0), 21, -1)[:, :, self.idx]
        dist_x = (x[:, :, :].argmax(1) != x0[:, :, :].argmax(1)).int().float().sum(-1)
        dist_y = (x[:, :, :].argmax(1) != x1[:, :, :].argmax(1)).int().float().sum(-1)
        return (dist_x / (dist_x + dist_y) - t / self.N).pow(2)


class SimCriterion(AbstractCriterion):
    """ Similarity criterion.

    Args:
        x0 (torch.Tensor): Input tensor source.
        nnz_idx (list): List of non-zero indices.
        postprocessing (function): Postprocessing function.
        name (str): Name of the criterion.
    """

    def __init__(self, x0, nnz_idx, postprocessing=None, name="sim"):
        super(SimCriterion, self).__init__(name)
        self.postprocessing = (lambda x: x)
        self.nnz_idx = nnz_idx
        self.x0 = x0.reshape(1, 21, -1)
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        """ Run the criterion.

        Args:
            x (torch.Tensor): Input tensor.
            Wv (torch.Tensor): Weight tensor.

        Returns:
            phi (torch.Tensor): Criterion value.
            gradient_phi (torch.Tensor): Criterion gradient.
        """

        samples, batch_size = x.size(0), x.size(1)
        sim = self.apply(x.reshape(samples * batch_size, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(sim)

        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        """ Apply the criterion.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Similarity criterion.
        """
        x0 = self.x0.reshape(1, 21, -1)
        x = x.reshape(x.size(0), 21, -1)
        return (x[:, :, self.nnz_idx].argmax(1) != x0[:, :, self.nnz_idx].argmax(1)).int().float().sum(-1)


class RbmCriterion(AbstractCriterion):
    """ Rbm criterion.

    Args:
        model (Rbm): Rbm model.
        postprocessing (function): Postprocessing function.
        name (str): Name of the criterion.
    """
    def __init__(self, model, postprocessing=None, name="erbm"):
        super(RbmCriterion, self).__init__(name)
        self.postprocessing = (lambda x: x)
        self.model = model
        self.Z = self.model.Z
        self.N = self.model.layers["pi"].N
        if postprocessing is not None:
            self.postprocessing = postprocessing

    def run(self, x, Wv):
        """ Run the criterion.

        Args:
            x (torch.Tensor): Input tensor.
            Wv (torch.Tensor): Weight tensor.

        Returns:
            phi (torch.Tensor): Criterion value.
            gradient_phi (torch.Tensor): Criterion gradient.
        """
        samples, batch_size = x.size(0), x.size(1)
        e = self.apply(x.reshape(samples, batch_size, -1)).to(device).reshape(samples, batch_size, -1)
        phi = self.postprocessing.run(e)
        gradient_phi = ((Wv * phi).mean(0) - phi.mean(0) * Wv.mean(0))
        return phi.mean(0)[:, 0], gradient_phi

    def apply(self, x):
        """ Apply the criterion.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rbm criterion.
        """
        samples, batch_size = x.size(0), x.size(1)
        return self.model({"pi": x.reshape(samples * batch_size, 21, N)[:, :, :]}) / 736 - self.Z
