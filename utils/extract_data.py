import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import torch
import numpy as np

from random import shuffle

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from utils import *

def from_fasta_to_df(folder, file, chunksize=5000):
    ids, seqs, aligned_seqs = [], [], []
    pd.DataFrame(columns=["aligned_seq", "seq", "length"]).to_csv(f"{folder}/sequences.csv")
    with open(file, "r") as input_handle:
        for i, seq in enumerate(SeqIO.parse(input_handle, "fasta")):
            seq = seq.upper()
            if len(ids) >= chunksize:
                df = pd.DataFrame(index=ids)
                df["aligned_seq"] = aligned_seqs
                df["aligned_seq"] = aligned_seqs
                df["seq"] = seqs
                df["length"] = df.seq.apply(lambda seq: len(seq))
                ids, seqs, aligned_seqs = [], [], []
                df.to_csv(f"{folder}/sequences.csv", mode="a", header=False)
            aligned_seq = str(seq.seq)
            if "X" in aligned_seq:
                continue
            ids.append(seq.id)
            seq = "".join([c for c in aligned_seq if c in AA])
            seqs.append(seq), aligned_seqs.append(aligned_seq)
            print(f"Processing {i} sequences ...", end="\r")
    df = pd.DataFrame(index=ids)
    df["aligned_seq"] = aligned_seqs
    df["seq"] = seqs
    df["length"] = df.seq.apply(lambda seq: len(seq))
    df.to_csv(f"{folder}/sequences.csv", mode="a", header=False)

def from_df_to_fasta(folder, df, prefix=""):
    records_aligned = []
    records_unaligned = []
    for i, data in enumerate(df.itertuples()):
        records_aligned.append(SeqRecord(Seq(data.pi_msa), id=str(data.id)))
        records_unaligned.append(SeqRecord(Seq(data.pi_msa), id=str(data.id)))
    with open(f"{folder}/{prefix}aligned.fasta", "w") as handle:
        SeqIO.write(records_aligned, handle, "fasta")
    with open(f"{folder}/{prefix}unaligned.fasta", "w") as handle:
        SeqIO.write(records_unaligned, handle, "fasta")


def from_df_to_data(folder, df, npam=10, prefix=""):
    pam = f"pam{npam}"
    N = len(df.pi_msa.values[0])
    all_pi = torch.zeros(len(df), 20, N)
    all_pam = torch.zeros(len(df), 4**npam)
    for i, (pi_msa, pam) in enumerate(zip(df.pi_msa.values, df[pam].values)):
        pi_ = torch.tensor([AA_IDS.get(aa, 20) for aa in pi_msa])
        pi_ = torch.tensor(to_onehot(pi_, (None, 21)))
        all_pi[i] = pi_.t()[:-1]
        all_pam[i] = torch.tensor(pam)

    data = {"pi_msa": all_pi, "pam": all_pam, "L": len(df)}
    torch.save(data, f"{folder}/{prefix}data.pt")


# CLUSTERING/WEIGHTING/SPLITING

def cluster_weights(folder):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["cluster", "id"]).set_index("id").cluster
    cluster_weights = 1 / clusters.value_counts()
    weights = [cluster_weights[c] for c in clusters]
    push(f"{folder}/data.pt", "cluster_index", list(clusters.index))
    push(f"{folder}/data.pt", "weights", torch.tensor(list(weights)))
    return pd.Series(data=weights, index=clusters.index)


def split_train_val_set(folder, ratio=0.1):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["clusters", "id"]).set_index(
        "id").clusters
    max_size = ratio * len(clusters)
    val = []
    unique_clusters = list(clusters.unique())
    shuffle(unique_clusters)
    for c in unique_clusters:
        val += list(clusters[clusters == c].index)
        if len(val) > max_size:
            break
    is_val = torch.tensor([int(c in val) for c in clusters.index])
    subset = dict()
    subset["val"] = torch.where(is_val == 1)[0]
    subset["train"] = torch.where(is_val == 0)[0]
    push(f"{folder}/data.pt", "subset", subset)
    return pd.Series(data=is_val, index=clusters.index)

# Motifs and Binding sites

def get_binding_sites(X):
    try:
        return [int(x) for x in X.split("|")[1].split("=")[1].split(",")[:-1]]
    except:
        return None


def dist_matrix(x, n=5):
    idxs = np.where(x == 1)[0]
    dist = np.zeros(x.shape[0] + 2 * n)

    for i in range(n, -1, -1):
        dist[idxs + n - i] = 1 / (i + 1)
        dist[idxs + n + i] = 1 / (i + 1)
    return dist[n:-n]