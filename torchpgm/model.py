import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

import time
import networkx as nx

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from utils import *

from .edge import Edge

DATA = '/home/malbrank/data'


class MRF(nn.Module):
    r"""
    Class to handle Markov Random Field : graph of layers and edges.
    Args:
        layers (Dict): Keys are the name of the layers, values are the layers
        edges (List of tuples): List of all edges between layers
    """

    def __init__(self, layers, edges, name, method="pcd"):
        super(MRF, self).__init__()
        self.layers = nn.ModuleDict(layers)
        self.in_, self.out_ = [k for k in self.layers.keys() if k != "hidden"], "hidden"
        self.edges_name = edges
        self.edges = nn.ModuleDict({f"{u} -> {v}": Edge(layers[u], layers[v],
                                                        ZeroSumGauge(layers[u].N, layers[u].q)).to(device) for u, v in
                                    edges})
        # self.G = self.build_graph()
        self.Z = 0
        # self.ais(n_inter=200)
        self.device = "cpu"
        self.name = f"{name}-{self.layers[self.out_].N}"
        self.writer = SummaryWriter(f"{DATA}/tensorboard/pgm/{self.name}")
        self.method = "pcd"

        # draw_G(self.G)

    def __repr__(self):
        return f"MRF {self.name}"

    def to(self, device):
        super(MRF, self).to(device)
        for edge in self.edges.values():
            edge.to(device)
        for layer in self.layers.values():
            layer.to(device)
        self.device = device
        return self

    def build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(list(self.layers.keys()))
        G.add_edges_from(self.edges_name)
        return G

    def get_edge(self, i, o):
        r"""

        :param i: str
        :param o: str
        :return: Edge
        """
        return self.edges[f"{i} -> {o}"]

    def is_edge(self, i, o):
        return f"{i} -> {o}" in self.edges.keys()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def save_model(self, filename):
        torch.save(self, filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def forward(self, d):
        # return self.integrate_likelihood(d, "hidden")
        return self.integrate_likelihood(d, "hidden")

    def gauge_weights(self):
        for _, edge in self.edges.items():
            edge.gauge_weights()
        for _, lay in self.layers.items():
            lay.gauge_weights()

    def integrate_likelihood(self, d, out_lay, beta=1):  # compute P(visible)
        out = self.layers[out_lay]
        ll = None
        for name, x in d.items():
            if name == out_lay:
                continue
            Ix = self.get_edge(name, out_lay)(x, False)
            gammaIx = out.gamma(beta * Ix)
            ll = gammaIx if ll is None else ll + gammaIx
            ll += self.layers[name](x).reshape(-1)
        return ll

    def full_likelihood(self, d, beta=1):
        e = None
        for (in_, out_), edge in zip(self.edges_name, self.edges.values()):
            if in_ not in d.keys() or out_ not in d.keys():
                continue
            x, h = d[in_], d[out_]
            e = beta * (edge(x, False) * h).sum(-1).view(-1) if e is None else e + beta * (edge(x, False) * h).sum(
                -1).view(-1)
        for in_, x in d.items():
            e += self.layers[in_](x).view(-1)
        return e

    def gibbs_sampling(self, d, in_lays, out_lays, k=1, beta=1):
        lays = in_lays + out_lays
        for lay in out_lays:
            d[lay] = self._gibbs(d, lay, beta)
        d_0 = d.copy()
        for _ in range(k):
            for lay in lays:
                d[lay] = self._gibbs(d, lay, beta)
        return d_0, d

    def init_sample(self, n_samples, beta=0):
        in_, out_ = self.in_, self.out_
        d = dict()
        out_layer = self.layers[out_]
        d[out_] = out_layer.sample([torch.zeros(n_samples, out_layer.N).to(self.device)], beta)
        for lay in in_:
            d[lay] = self._gibbs(d, lay, beta)
        return d

    def ais(self, n_samples=20, n_inter=200, verbose=False):
        in_, out_ = self.in_, self.out_
        d = self.init_sample(n_samples)
        betas = torch.linspace(0, 1, n_inter)
        weights = 0
        for i, (last_beta, beta) in enumerate(zip(betas[:-1], betas[1:])):
            _, d_f = self.gibbs_sampling(d, in_, [out_], 1, beta)
            arate = torch.exp(self.full_likelihood(d_f, beta) - self.full_likelihood(d, beta))
            accepted_idx = torch.where(torch.rand_like(arate) < arate)[0]
            for lay in d.keys():
                d[lay][accepted_idx] = d_f[lay][accepted_idx]
            weights += (self.full_likelihood(d, beta) - self.full_likelihood(d, last_beta)).mean(0).cpu().item()
            if verbose and not i % (n_inter // 10):
                print(f"Iteration {i} : {weights}")
        Z_0 = sum([layer.Z_0() for layer in self.layers.values()])
        Z = (weights + Z_0) / sum(self.layers[i].N for i in in_)
        if verbose:
            print(f"Estimated Z : {Z:.3f}")
        self.Z = Z
        return Z.cpu().item()

    def _gibbs(self, d, out_lay, beta=1):
        probas = []
        out_layer = self.layers[out_lay]
        for name, lay in d.items():
            if name == out_lay:
                continue
            distribution = self._distribution(name, out_lay, lay)
            if distribution is not None:
                probas.append(distribution)
        return out_layer.sample(probas, beta)

    def _distribution(self, i, o, x):
        if self.is_edge(i, o):
            return self.get_edge(i, o).forward(x, sample=False)
        if self.is_edge(o, i):
            return self.get_edge(o, i).backward(x, sample=False)
        return None

    def write_tensorboard(self, logs, n_iter):
        for k, v in logs.items():
            self.writer.add_scalar(k, v, n_iter)

    def raise_sample_through_criterion(self, x_0, criterion, active_visible, inactive_units=[], target=5, T=1e-5,
                                       n_sampling=1000):
        edge = self.edges["pi -> hidden"]
        batch_size, q, N = x_0.size()
        x_0 = x_0.view(batch_size, -1).float()
        Z = self.Z
        state_e, state_mut, state_x, state_p = self({"pi": x_0.float().to(self.device)}) / (
                    2 * N) - Z, x_0.clone(), x_0.clone(), criterion(x_0.clone())
        x_chains = [[x_] for x_ in x_0]
        e_chains = [[e_.cpu().item()] for e_ in state_e]
        n_mut = 0
        for i in range(1, n_sampling):
            x_next, _, mut, _ = edge.partial_gibbs_step(state_x, active_visible, inactive_units)
            x_next = x_next.float()
            e = self({"pi": x_next.float().clone()}) / N - Z
            p = criterion(x_next.clone())

            idx = torch.where((torch.rand(batch_size) < torch.exp(1 / T * (p - state_p))))[0]
            n_mut += len(idx)

            state_x[idx] = x_next[idx].float()
            state_mut[idx] = mut[idx]
            state_p[idx] = p[idx]
            state_e[idx] = e[idx]
            for i_ in idx:
                x_chains[i_].append(mut[i_].detach())
                e_chains[i_].append(e[i_].cpu().item())
            print(
                f"{n_mut}/{16 * i} [{(100 * n_mut) / (16 * i):.2f}%] || Class : {state_p.exp().mean().cpu().item():.3f} || E = {state_e.mean():.3f}",
                end="\r")
        return state_x, state_mut, x_chains, e_chains

    def train_epoch(self, optimizer, loader, visible_layers, hidden_layers, gammas, epoch, savepath="seq100"):
        pass

    def val(self, loader, visible_layers, hidden_layers, epoch):
        pass


class PAM_classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.8):
        super(PAM_classifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear1(self.bn1(x))

    def l1b_reg(self):
        weights1 = self.linear1.weight.data
        return weights1.pow(2).sum(0).sum(0)


class PI_RBM(MRF):
    def __repr__(self):
        return f"PI MRF {self.name}"

    def train_epoch(self, optimizer, loader, dataset,
                    regularizer=dict(),
                    epoch=0, savepath="seq100"):
        start = time.time()
        self.train()
        mean_loss, mean_reg, mean_acc_pi, mean_pv, mean_pvh = 0, 0, 0, 0, 0
        m, s = 0, 0
        self.l1b, self.l2 = l1b, l2 = regularizer.get("l1b", None), regularizer.get("l2", None)
        edge = self.get_edge("pi", "hidden")
        pi, h = self.layers["pi"], self.layers["hidden"]
        visible_layers, hidden_layers = ["pi"], ["hidden"]
        for batch_idx, (x_d, x_m, w, idxs) in enumerate(loader):
            d_0 = {"pi": x_d.float().to(self.device)}
            d_f = {"pi": x_m.float().to(self.device)}

            w = w.float().to(self.device)
            batch_size, qpi, Npi = d_0["pi"].size()

            # Sampling
            _, d_f = self.gibbs_sampling(d_f, ["pi"], ["hidden"], k=10)
            d_f = {k: v.detach() for k, v in d_f.items()}
            if self.method == "pcd":
                dataset.update_pcd(idxs, d_f["pi"].detach().cpu())

            # Optimizationby increasing number of weight vectors to 512 or more. But I encountered this problem several times earlier with different RBM types (binary, Gaussian, even convolutional), different number of hidden units (including pretty large), different hyper-parameters, etc.
            optimizer.zero_grad()
            e_0, e_f = self(d_0), self(d_f)
            loss = msa_mean(e_f - e_0, w) / Npi
            reg = torch.tensor(0., device=self.device)
            if l1b is not None:
                reg += l1b / pi.shape * edge.l1b_reg()
            if l2 is not None:
                reg += l2 / pi.shape * pi.l2_reg()
            loss = loss + reg
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 10)
            optimizer.step()
            self.gauge_weights()
            self.layers["hidden"].update_params()

            # Metrics
            d_0, d_1 = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
            acc_pi = aa_acc(d_0["pi"].reshape(batch_size, qpi, Npi), d_1["pi"].reshape(batch_size, qpi, Npi))
            pv = msa_mean(self.integrate_likelihood(d_0, "hidden"), w) / Npi - self.Z
            pvh = msa_mean(self.full_likelihood(d_0), w) / Npi - self.Z
            mean_pv = (mean_pv * batch_idx + pv.cpu().item()) / (batch_idx + 1)
            mean_pvh = (mean_pvh * batch_idx + pvh.cpu().item()) / (batch_idx + 1)
            mean_loss = (mean_loss * batch_idx + loss.cpu().item()) / (batch_idx + 1)
            mean_reg = (mean_reg * batch_idx + reg.cpu().item()) / (batch_idx + 1)
            mean_acc_pi = (mean_acc_pi * batch_idx + acc_pi) / (batch_idx + 1)
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60

            print(
                f'''Train Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || P(v): {mean_pv:.3f} || Reg: {mean_reg:.3f} || PI: {mean_acc_pi:.3f}''',
                end="\r")
        print(
            f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || P(v): {mean_pv:.3f} || Reg: {mean_reg:.3f} || PI: {mean_acc_pi:.3f}''',
        )
        logs = {"train/loss": mean_loss, "train/reg": mean_reg, "train/acc_pi": mean_acc_pi}
        self.write_tensorboard(logs, epoch)
        if not epoch % 30:
            self.save(f"{savepath}_{epoch}.h5")

    def val(self, loader, visible_layers, hidden_layers, epoch):
        start = time.time()
        self.eval()
        mean_pv, mean_pvh, mean_reg, mean_acc_pi = 0, 0, 0, 0
        m, s = 0, 0
        self.ais()
        edge = self.get_edge("pi", "hidden")
        pi, h = self.layers["pi"], self.layers["hidden"]
        for batch_idx, (x_d, x_m, w, idxs) in enumerate(loader):
            d_0 = {visible_layers[0]: x_d.float().to(self.device)}
            w = w.float().to(self.device)
            batch_size, qpi, Npi = d_0["pi"].size()
            N = Npi

            # Sampling
            d_0, d_1 = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)

            acc_pi = aa_acc(d_0["pi"].reshape(batch_size, qpi, Npi), d_1["pi"].reshape(batch_size, qpi, Npi))

            pv = msa_mean(self.integrate_likelihood(d_0, "hidden"), w) / N - self.Z
            pvh = msa_mean(self.full_likelihood(d_0), w) / N - self.Z
            mean_pv = (mean_pv * batch_idx + pv.cpu().item()) / (batch_idx + 1)
            mean_pvh = (mean_pvh * batch_idx + pvh.cpu().item()) / (batch_idx + 1)
            mean_acc_pi = (mean_acc_pi * batch_idx + acc_pi) / (batch_idx + 1)
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
            print(
                f'''Val Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} || P(v): {mean_pv:.3f} || P(v,h): {mean_pvh:.3f} || PI: {mean_acc_pi:.3f}''',
                end='\r')

        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        logs = {"val/p(v)": mean_pv, "val/p(v,h)": mean_pvh, "val/acc_pi": mean_acc_pi, "val/ais": self.Z}
        self.write_tensorboard(logs, epoch)
        print(
            f'''Val Epoch: {epoch} [100%] || Time: {m} min {s} || P(v): {mean_pv:.3f} || P(v,h): {mean_pvh:.3f} || PI: {mean_acc_pi:.3f}''')
        return mean_pv


class PI_RBM_SSL(PI_RBM):
    def __init__(self, classifier, layers, edges, name, gamma=1, method="pcd"):
        super(PI_RBM_SSL, self).__init__(layers, edges, name, method)
        self.classifier = classifier
        self.gamma = gamma

    def __repr__(self):
        return f"PI MRF SSL {self.name}"

    def train_epoch_classifier(self, optimizer, loader1, loader2, dataset1, dataset2,
                               regularizer=dict(),
                               epoch=0, savepath="seq100"):
        start = time.time()
        self.train()
        mean_loss, mean_reg, mean_acc_pi, mean_acc_pam, mean_pvh, mean_pv = 0, 0, 0, 0, 0, 0
        m, s = 0, 0
        self.l1b, self.l2, self.pam_l1b = l1b, l2, pam_l1b = regularizer.get("l1b", None), regularizer.get("l2",
                                                                                                           None), regularizer.get(
            "pam_l1b", None)
        edge = self.get_edge("pi", "hidden")
        pi, h = self.layers["pi"], self.layers["hidden"]
        visible_layers, hidden_layers = ["pi"], ["hidden"]
        for batch_idx, ((x_d, x_m, w, idxs), (x_dd, x_mm, pam, ww, idxss)) in enumerate(zip(loader1, loader2)):
            d_00 = {"pi": x_dd.float().to(self.device)}
            pam = pam.float().to(self.device)
            w = w.float().to(self.device)
            batch_sizee, qpi, Npi = d_00["pi"].size()
            pam_pred_0 = self.classifier(edge(d_00["pi"], ))

            d_0 = {"pi": x_d.float().to(self.device)}
            d_f = {"pi": x_m.float().to(self.device)}
            w = w.float().to(self.device)
            batch_size, qpi, Npi = d_0["pi"].size()

            # Sampling
            _, d_f = self.gibbs_sampling(d_f, ["pi"], ["hidden"], k=10)
            d_f = {k: v.detach() for k, v in d_f.items()}
            if self.method == "pcd":
                dataset1.update_pcd(idxs, d_f["pi"].detach().cpu())

            optimizer.zero_grad()
            e_0, e_f = self(d_0), self(d_f)
            loss = msa_mean(e_f - e_0, w) / Npi
            pam_loss = F.binary_cross_entropy(F.sigmoid(pam_pred_0).flatten(), pam.flatten()) / batch_sizee
            reg = torch.tensor(0., device=self.device)
            if pam_l1b is not None:
                reg += pam_l1b * self.classifier.l1b_reg()
            if l1b is not None:
                reg += l1b / pi.shape * edge.l1b_reg()
            if l2 is not None:
                reg += l2 / pi.shape * pi.l2_reg()
            loss = (self.gamma * pam_loss + loss) + reg
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 10)
            optimizer.step()
            self.gauge_weights()
            self.layers["hidden"].update_params()

            # Metrics
            d_0, d_1 = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
            pv = msa_mean(self.integrate_likelihood(d_0, "hidden"), w) / Npi - self.Z
            pvh = msa_mean(self.full_likelihood(d_0), w) / Npi - self.Z
            mean_pv = (mean_pv * batch_idx + pv.cpu().item()) / (batch_idx + 1)
            mean_pvh = (mean_pvh * batch_idx + pvh.cpu().item()) / (batch_idx + 1)

            acc_pi = aa_acc(d_0["pi"].reshape(batch_size, qpi, Npi), d_1["pi"].reshape(batch_size, qpi, Npi))
            acc_pam = [roc_auc_score(pam_.int().cpu(), pam_pred_.cpu()) for pam_, pam_pred_ in
                       zip(pam.t(), pam_pred_0.detach().t()) if ((sum(pam_) > 0) and (sum(pam_) < len(pam_)))]
            try:
                acc_pam = sum(acc_pam) / len(acc_pam)
            except:
                acc_pam = .5
            mean_loss = (mean_loss * batch_idx + loss.cpu().item()) / (batch_idx + 1)
            mean_reg = (mean_reg * batch_idx + reg.cpu().item()) / (batch_idx + 1)
            mean_acc_pi = (mean_acc_pi * batch_idx + acc_pi) / (batch_idx + 1)
            mean_acc_pam = (mean_acc_pam * batch_idx + acc_pam) / (batch_idx + 1)

            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60

            print(
                f'''Train SSL Epoch: {epoch} [{int(100 * batch_idx / len(loader1))}%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || P(v): {mean_pv:.3f} || Reg: {mean_reg:.3f} || PI: {mean_acc_pi:.3f} || PAM: {mean_acc_pam:.3f}''',
                end="\r")
        print(
            f'''Train SSL Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || P(v): {mean_pv:.3f} || Reg: {mean_reg:.3f} || PI: {mean_acc_pi:.3f}  || PAM: {mean_acc_pam:.3f}''',
        )
        logs = {"train/acc_pam": mean_acc_pam}
        self.write_tensorboard(logs, epoch)
        if not epoch % 30:
            self.save(f"{savepath}_{epoch}.h5")

    def val_classifier(self, loader, visible_layers, hidden_layers, epoch):
        start = time.time()
        self.eval()
        mean_pv, mean_pvh, mean_reg, mean_acc_pi, mean_acc_pam = 0, 0, 0, 0, 0
        edge = self.get_edge("pi", "hidden")
        pi, h = self.layers["pi"], self.layers["hidden"]
        for batch_idx, (x_d, x_m, pam, w, idxs) in enumerate(loader):
            d_0 = {visible_layers[0]: x_d.float().to(self.device)}
            pam = pam.float().to(self.device)
            w = w.float().to(self.device)
            batch_size, qpi, Npi = d_0["pi"].size()
            N = Npi

            # Sampling
            d_0, d_1 = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
            pam_pred = self.classifier(edge(d_0["pi"], False))

            acc_pi = aa_acc(d_0["pi"].reshape(batch_size, qpi, Npi), d_1["pi"].reshape(batch_size, qpi, Npi))
            acc_pam = [roc_auc_score(pam_.int().cpu(), pam_pred_.cpu()) for pam_, pam_pred_ in
                       zip(pam.t(), pam_pred.detach().t()) if ((sum(pam_) > 0) and (sum(pam_) < len(pam_)))]
            acc_pam = sum(acc_pam) / len(acc_pam)

            pv = msa_mean(self.integrate_likelihood(d_0, "hidden"), w) / N - self.Z
            pvh = msa_mean(self.full_likelihood(d_0), w) / N - self.Z
            mean_pv = (mean_pv * batch_idx + pv.cpu().item()) / (batch_idx + 1)
            mean_pvh = (mean_pvh * batch_idx + pvh.cpu().item()) / (batch_idx + 1)
            mean_acc_pi = (mean_acc_pi * batch_idx + acc_pi) / (batch_idx + 1)
            mean_acc_pam = (mean_acc_pam * batch_idx + acc_pam) / (batch_idx + 1)
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
            print(
                f'''Val SSL Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} || P(v): {mean_pv:.3f} || P(v,h): {mean_pvh:.3f} || PI: {mean_acc_pi:.3f} || PAM: {mean_acc_pam:.3f}''',
                end="\r")

        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        logs = {"val/acc_pam": mean_acc_pam}
        self.write_tensorboard(logs, epoch)
        print(
            f'''Val SSL Epoch: {epoch} [100%] || Time: {m} min {s} || P(v): {mean_pv:.3f} || P(v,h): {mean_pvh:.3f} || PI: {mean_acc_pi:.3f} || PAM: {mean_acc_pam:.3f}''')
