from itertools import product
import torch
import numpy as np

NAd_idx = {"A":[1,0,0,0], "T":[0,1,0,0], "C":[0,0,1,0], "G":[0,0,0,1],
          "W":[1,1,0,0], "S":[0,0,1,1], "M":[1,0,1,0], "K":[0,1,0,1], "R":[1,0,0,1], "Y":[0,1,1,0],
           "B":[0,1,1,1], "D":[1,1,0,1], "H":[1,1,1,0], "V": [1,0,1,1], "N":[1,1,1,1]}
NAd_in = {"A":"A", "T":"T", "C":"C", "G":"G",
          "W":"AT", "S":"CG", "M":"AC", "K":"TG", "R":"AG", "Y":"TC",
           "B":"TCG", "D":"ATG", "H":"ATC", "V": "ACG", "N":"ATCG"}
NAd_tensor = {k:torch.tensor(v) for k,v in NAd_idx.items()}

ONEHOT = [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]

PAM3 = [x+y+z for x,y,z in product("ATCG", "ATCG", "ATCG")]
PAM3_tensor = torch.tensor([[x,y,z] for x,y,z in product(ONEHOT,ONEHOT,ONEHOT)]).int().permute(0,2,1)
PAM3_idx = {k:v for v, k in enumerate(PAM3)}

PAM4 = [x+y+z+t for x,y,z,t in product("ATCG", "ATCG", "ATCG", "ATCG")]
PAM4_tensor = torch.tensor([[x,y,z,t] for x,y,z,t in product(ONEHOT,ONEHOT,ONEHOT,ONEHOT)]).int().permute(0,2,1)
PAM4_idx = {k:v for v, k in enumerate(PAM4)}

PAM9 = [x1+x2+x3+x4+x5+x6+x7+x8+x9 for x1,x2,x3,x4,x5,x6,x7,x8,x9 in product("ATCG", "ATCG", "ATCG","ATCG", "ATCG", "ATCG", "ATCG", "ATCG", "ATCG")]
PAM9_tensor = torch.tensor([[xi for xi in x] for x in product(ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT)]).int().permute(0,2,1)
PAM9_idx = {k:v for v, k in enumerate(PAM9)}

PAM10 = [x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 for x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 in product("ATCG", "ATCG", "ATCG", "ATCG","ATCG", "ATCG", "ATCG", "ATCG", "ATCG", "ATCG")]
PAM10_tensor = torch.tensor([[xi for xi in x] for x in product(ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT,ONEHOT)]).int().permute(0,2,1)
PAM10_idx = {k:v for v, k in enumerate(PAM10)}

def onehot_pam3(x, vec):
    vec[PAM3_idx[x]] = 1
    return vec

def onehot_pam4(x):
    pam4 = np.zeros(256, dtype=int)
    pam4[PAM4_idx[x]] = 1
    return pam4

def onehot_pam9(x):
    pam9 = np.zeros(len(PAM10), dtype=int)
    pam9[PAM9_idx[x]] = 1
    return pam9

def onehot_pam10(x):
    pam10 = np.zeros(len(PAM10), dtype=int)
    pam10[PAM10_idx[x]] = 1
    return pam10
