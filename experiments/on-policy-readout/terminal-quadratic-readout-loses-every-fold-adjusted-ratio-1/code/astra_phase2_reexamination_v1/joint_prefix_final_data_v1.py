"""Manifest-backed target-blind loader and unchanged MPS card constructor."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
PKEYS = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b',
         'head','head_b','gain')
QKEYS = ('w0','b0','w2','b2','w4','b4')
assert len(TRAIN)==160 and len(VALID)==40 and not set(TRAIN)&set(VALID)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_path(rank):
    manifest = json.loads((STORE/'manifest.json').read_text())
    items = manifest['roles']['observer_fit']['row_manifests']
    matching = [item for item in items if item['role_rank']==rank]
    assert len(matching)==1
    item = matching[0]
    row = STORE/item['path']
    assert sha(row)==item['sha256']
    path = row.parent/'weights.npy'
    assert path.is_file()
    return path


def fixed_features(weights):
    depth,n,_ = weights.shape
    fixed = np.zeros((depth,n,30),np.float32)
    # Same NumPy expressions and reduction axes as frozen FullCard.__init__.
    w2 = weights*weights
    w3,w4 = w2*weights,w2*w2
    fixed[:,:,16] = weights.sum(1)*.7071067811865475
    fixed[:,:,17] = (w2-.001953125).sum(1)*11.313708498984761
    fixed[:,:,18] = w3.sum(1)*93.47798314754834
    fixed[:,:,19] = (w4-1.1444091796875e-5).sum(1)*836.0924988699916
    return fixed


def original_payloads():
    with np.load(HERE/'payload_compact_feed_fullpair_v1.npz') as src:
        p = {k:src[k].copy() for k in src.files}
    with np.load(HERE/'payload_fullpair_final_head_v1.npz') as src:
        q = {k:src[k].copy() for k in src.files}
    return p,q


class ManifestFullCard(ExplicitBoundBankFullCard):
    def __init__(self,rank,P,Q,audit_selections=False):
        self.device = torch.device('mps')
        self.dtype = torch.float32
        wn = np.load(weight_path(rank),mmap_mode='r')
        assert wn.shape==(16,1024,1024) and wn.dtype==np.float32
        self.n,self.depth = wn.shape[1],len(wn)
        self.w = torch.tensor(wn,device=self.device,dtype=torch.float32)
        self.ws = self.w
        self.fixed = torch.tensor(fixed_features(wn),device=self.device,dtype=self.dtype)
        self.P0,self.Q0 = P,Q
        self.audit_selections = audit_selections
        self.selections = []
