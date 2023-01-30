import warnings
warnings.filterwarnings("ignore")

from random import shuffle

from torch import optim

from torchpgm import *
from config import *

device = "cuda"
folder = f"{DATA}/vink"

batch_size = 300
Nh = 200
Npam = 5
n_epochs = 1000
start_supervision_epoch = 20
l1b = 0.0025
l2 = 0.05
lambda_pa = 0.00
lr = 0.0001
gammas = sorted(gammas)[260:542]
model_full_name = f"rbmssl_pid_h{Nh}_npam{Npam}"

for gamma in gammas:
    torch.cuda.empty_cache()
    train_dataset = RBMData(f"{folder}/data.pt", subset="train")
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)

    val_dataset = RBMData(f"{folder}/data.pt", subset="val")
    val_loader = DataLoader(val_dataset, batch_size=300, shuffle=False)

    train_dataset_labelled = RBMDataWithPAM(f"{folder}/data.pt", Npam, subset="train_labelled")
    train_loader_labelled = DataLoader(train_dataset_labelled, batch_size=11, shuffle=True, drop_last=True)

    val_dataset_labelled = RBMDataWithPAM(f"{folder}/data.pt", Npam, subset="val_labelled")
    val_loader_labelled = DataLoader(val_dataset_labelled, batch_size=300, shuffle=False)

    q_pi, N_pi = train_dataset[0][0].size()
    g_pi = torch.zeros(q_pi, N_pi)
    for x_pi, _, w, _ in train_dataset:
        g_pi += w * x_pi
    g_pi = np.log(1 + g_pi)
    W = sum(train_dataset.weights)
    g_pi = (g_pi - g_pi.mean(0)[None]).flatten() / W

    device = "cuda"
    visible_layers = ["pi"]
    hidden_layers = ["hidden"]

    pi = OneHotLayer(None, N=N_pi, q=q_pi, name="pi")
    h = GaussianLayer(N=Nh, name="hidden")
    classifier = PAM_classifier(Nh, Npam * 4, dropout=0.)
    E = [(pi.name, h.name)]
    E.sort()

    model_rbm = PI_RBM_SSL(classifier, layers={pi.name: pi, h.name: h}, edges=E, gamma=gamma,
                           name=f"{model_full_name}_gamma{gamma}")
    optimizer = optim.AdamW(model_rbm.parameters(), lr=lr)
    model_rbm = model_rbm.to("cuda")
    model_rbm.ais()

    for epoch in range(1, 181):
        model_rbm.train_epoch_classifier(optimizer, train_loader, train_loader_labelled, train_dataset,
                                         train_dataset_labelled,
                                         regularizer={"l1b": l1b, "l2": l2, "l1b_pam": 0},
                                         epoch=epoch, savepath=f"{folder}/weights/{model_full_name}_gamma{gamma}")
        shuffle(train_dataset_labelled.x_m)
        shuffle(train_dataset.x_m)

        if not epoch % 5:
            model_rbm.ais()
            model_rbm.val(val_loader, visible_layers, hidden_layers, epoch)
            model_rbm.val_classifier(val_loader_labelled, visible_layers, hidden_layers, epoch)