from torchpgm.model import *
from torchpgm.layers import *

from cld.postprocessing import *
from cld.criterion import *
from cld.walker import *

from .utils import *
from .config import *

device = "cuda"
folder = f"{DATA}/vink"
Nh, Npam = 200, 5
best_epoch = 90
q_pi, N_pi = 21, 736
model_full_name = f"rbmssl2_pid_h{Nh}_npam{Npam}_gamma8.819763977946266"

def lit_to_pam(s):
    pam = []
    s += "N" * max(0, (Npam - len(s)))
    for x in s:
        pam += NAd_idx[x]
    return torch.tensor(pam).float()[None].to(device)


pi = OneHotLayer(None, N=N_pi, q=q_pi, name="pi")
h = GaussianLayer(N=Nh, name="hidden")
classifier = PAM_classifier(Nh, Npam * 4)
E = [(pi.name, h.name)]
E.sort()

model_rbm = PI_RBM_SSL(classifier, layers={pi.name: pi, h.name: h}, edges=E, name=model_full_name)
model_rbm = model_rbm.to(device)
model_rbm.load(f"{folder}/weights/{model_full_name}_{best_epoch}.h5")
model_rbm.eval()
model_rbm = model_rbm.to("cpu")
model_rbm.ais()

model_rbm = model_rbm.to("cpu")

x_cas9 = torch.load(f"{DATA}/x_cas9.pt")
zero_idx = torch.load(f"{DATA}/zero_idx.pt")
kept_idx = torch.load(f"{DATA}/kept_idx.pt")
target = lit_to_pam("NGG")

objective = RbmCriterion(model_rbm, postprocessing=ConstantPostprocessor(0))
constraints = [
    SimCriterion(x_cas9, nnz_idx, postprocessing=ConstantPostprocessor(None, 80)),
    RbmCriterion(model_rbm, postprocessing=ConstantPostprocessor(-0.5, None)),
]

T = 0.3 * torch.ones(1, 1, len(kept_idx))

walker = Walker(x_cas9.view(21, -1).clone(), model_rbm, objective, constraints, zero_idx, gamma=1, n=1, a=1,
                c=1e-2, eps=1, target=target.cpu(), T=T, kept_idx=kept_idx)
