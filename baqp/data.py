"""Local files only. No download; test data never used in allocation."""
import pickle
from pathlib import Path
import numpy as np
import torch

CACHE = {'cifar10':'cifar10_edm','cifar100':'cifar100_styleganxl',
         'fmnist':'fmnist_edm_oracle_6000pc_seed42'}


def load_real(root, dataset):
    root=Path(root)
    if dataset=='fmnist':
        root=root/'FashionMNIST'/'raw'
        def read(prefix):
            x=np.fromfile(root/(prefix+'-images-idx3-ubyte'),dtype=np.uint8,offset=16).reshape(-1,1,28,28)
            y=np.fromfile(root/(prefix+'-labels-idx1-ubyte'),dtype=np.uint8,offset=8).astype(np.int64)
            return torch.from_numpy(x),torch.from_numpy(y)
        return (*read('train'),*read('t10k'))
    root=root/('cifar-10-batches-py' if dataset=='cifar10' else 'cifar-100-python')
    def read(names):
        xs,ys=[],[]
        for name in names:
            with open(root/name,'rb') as f: d=pickle.load(f,encoding='bytes')
            xs.append(d[b'data'].reshape(-1,3,32,32)); ys.extend(d[b'labels' if dataset=='cifar10' else b'fine_labels'])
        return torch.from_numpy(np.concatenate(xs)),torch.tensor(ys,dtype=torch.long)
    train=[f'data_batch_{i}' for i in range(1,6)] if dataset=='cifar10' else ['train']
    return (*read(train),*read(['test_batch' if dataset=='cifar10' else 'test']))


def load_synthetic(root,dataset,classes):
    path=Path(root)/CACHE[dataset]/'tensor_cache.pt'
    d=torch.load(path,map_location='cpu',weights_only=True)
    x,y=d['images'],d['labels'].long()
    if x.dtype!=torch.uint8 or x.ndim!=4 or len(x)!=len(y): raise ValueError('Invalid cache')
    if set(y.unique().tolist())!=set(range(classes)): raise ValueError('Missing/invalid synthetic classes')
    return x,y,path


def split_train(labels,classes,clients=6,alpha=.3,seed=2026,val_fraction=.1):
    rng=np.random.default_rng(seed); labels=np.asarray(labels)
    train=[]; valid=[]
    for c in range(classes):
        ids=rng.permutation(np.flatnonzero(labels==c)); v=max(1,int(len(ids)*val_fraction))
        valid.extend(ids[:v]); train.extend(ids[v:])
    train=np.array(train,dtype=np.int64)
    for _ in range(100):
        parts=[[] for _ in range(clients)]
        for c in range(classes):
            ids=rng.permutation(train[labels[train]==c]); probs=rng.dirichlet(np.full(clients,alpha))
            bins=np.split(ids,np.cumsum(rng.multinomial(len(ids),probs))[:-1])
            for k,idsk in enumerate(bins): parts[k].extend(idsk)
        if min(map(len,parts))>=32: break
    else: raise ValueError('Cannot construct nonempty split')
    parts=[np.array(p,dtype=np.int64) for p in parts]
    counts=np.array([np.bincount(labels[p],minlength=classes) for p in parts])
    return parts,np.array(valid,dtype=np.int64),counts


def audit(x,y,sx,sy,parts,classes):
    """Train-only pixel class-mean discrepancy; diagnostic, not gradient bound."""
    ids=np.concatenate(parts)
    means=[]
    for c in range(classes):
        real_ids=ids[np.asarray(y)[ids]==c]
        gen_ids=torch.where(sy==c)[0]
        # Chunk sums avoid allocating complete float32 copies of the caches.
        def mean(images,indices):
            total=torch.zeros(images.shape[1:],dtype=torch.float64)
            for chunk in np.array_split(np.asarray(indices),max(1,(len(indices)+511)//512)):
                if len(chunk): total+=images[torch.from_numpy(chunk)].double().sum(0)
            return total/len(indices)/255
        means.append(float((mean(x,real_ids)-mean(sx,gen_ids)).square().mean().sqrt()))
    return dict(synthetic_count=len(sx),synthetic_class_counts=torch.bincount(sy,minlength=classes).tolist(),
                class_mean_pixel_rmse=means,meaning='train-only diagnostic; not a uniform gradient residual certificate')
