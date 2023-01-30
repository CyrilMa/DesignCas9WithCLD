import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import torch, torchvision
from torch.utils.data import DataLoader, Dataset

class RBMData(Dataset):
    def __init__(self, file, subset=None, nnz_idx=None, z_idx = None):
        super(RBMData, self).__init__()
        data = torch.load(file)
        keys = list(data.keys())
        print("Available : ", *keys)
        if subset is not None:
            idx = data["subset"][subset]
        else:
            idx = torch.arange(data["L"])
        self.x_d = torch.stack(list(data["x"][idx]),0) if "x" in keys else None
        if z_idx is not None:
            self.x_d = self.x_d[torch.where((self.x_d.sum(1)[:,z_idx].mean(-1) < 0.1))[0]]
        if nnz_idx is not None:
            self.x_d = self.x_d[:,:,nnz_idx]
        self.x_d = list(self.x_d)
        self.L = len(self.x_d)
        self.weights = data["weights"][idx] if "weights" in keys else [1. for _ in self.x_d]
        self.x_m = []
        for i, x_d in enumerate(self.x_d):
            gaps = (x_d.sum(0) == 0).int()
            self.x_m.append(torch.cat([gaps[None], x_d], 0))
            self.x_d[i] = torch.cat([gaps[None], self.x_d[i]], 0)

    def update_pcd(self, idx, samples):
        for i, sample in zip(idx, samples):
            self.x_m[i] = sample

    def __len__(self):
        return self.L

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.weights[i], i

class RBMDataWithPAM(RBMData):
    def __init__(self, file, npam = 4, subset=None, nnz_idx=None, z_idx = None):
        data = torch.load(file)
        keys = list(data.keys())
        print("Available : ", *keys)
        if subset is not None:
            idx = data["subset"][subset]
        else:
            idx = torch.arange(data["L"])

        X_pams = data["y"][idx] if "y" in keys else None
        self.y = torch.stack([pam[:5].flatten() if pam is not None else None for pam in X_pams ],0)
        #for pam in PAM4:
        #    self.pams.append(torch.tensor(test_accept(X_pams, pam)))
        #self.pams = torch.cat([x[None] for x in self.pams],0).t()
        self.x_d = torch.stack(list(data["x"][idx]),0) if "x" in keys else None
        if z_idx is not None:
            self.x_d = self.x_d[torch.where((self.x_d.sum(1)[:,z_idx].mean(-1) <0.1))[0]]
            self.y = self.y[torch.where((self.x_d.sum(1)[:,z_idx].mean(-1) <0.1))[0]]
        if nnz_idx is not None:
            self.x_d = self.x_d[:,:,nnz_idx]
        self.x_d = list(self.x_d)
        self.L = len(self.x_d)

        self.weights = data["weights"][idx] if "weights" in keys else None
        self.x_m = []
        for i, x_d in enumerate(self.x_d):
            gaps = (x_d.sum(0) == 0).int()
            self.x_m.append(torch.cat([gaps[None], x_d], 0))
            self.x_d[i] = torch.cat([gaps[None], self.x_d[i]], 0)

    def __getitem__(self, i):
        return self.x_d[i], self.x_m[i], self.y[i], self.weights[i], i
