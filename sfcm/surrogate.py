"""
sfcm/surrogate.py  — dual-backend (PyTorch when available, sklearn fallback)
"""
from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import torch, torch.nn as nn, copy as _copy
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.neural_network import MLPRegressor as _SMLP
    _SK = True
except ImportError:
    _SK = False

if _TORCH:
    class _TorchMLP(nn.Module):
        def __init__(self, d_in, hidden, dropout):
            super().__init__()
            layers, in_d = [], d_in
            for h in hidden:
                layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
                in_d = h
            layers.append(nn.Linear(in_d, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

class EnsembleMLP:
    """
    Ensemble of n_members models.  sigma_hat(x) = std{y_k(x)}.
    Architecture default: [d,128,64,32,1], Dropout=0.1, ensemble=5  (Appendix B.1).
    Trains with PyTorch when available, sklearn MLPRegressor otherwise.
    """
    def __init__(self, d_in, hidden=(128,64,32), dropout=0.1, n_members=5,
                 lr=1e-3, patience=20, val_split=0.20, device=None, seed=0):
        self.d_in=d_in; self.hidden=hidden; self.dropout=dropout
        self.n_members=n_members; self.lr=lr; self.patience=patience
        self.val_split=val_split; self.seed=seed
        self._x_mean=np.zeros(d_in); self._x_std=np.ones(d_in)
        self._y_mean=0.0; self._y_std=1.0; self._fitted=False; self._members=[]
        self._dev = torch.device(device or ("cuda" if _TORCH and torch.cuda.is_available() else "cpu")) if _TORCH else None

    def fit(self, X, y):
        if len(X) < 2: return
        Xa = np.stack([np.asarray(x,dtype=np.float64) for x in X])
        ya = np.asarray(y, dtype=np.float64)
        self._x_mean = Xa.mean(0)
        self._x_std  = np.where(Xa.std(0)>1e-8, Xa.std(0), 1.0)
        self._y_mean = float(ya.mean())
        self._y_std  = float(ya.std()) if ya.std()>1e-8 else 1.0
        Xn = (Xa-self._x_mean)/self._x_std
        yn = (ya-self._y_mean)/self._y_std
        if _TORCH:   self._fit_torch(Xn, yn)
        elif _SK:    self._fit_sklearn(Xn, yn)
        else: raise ImportError("Install torch or scikit-learn.")
        self._fitted = True

    def _fit_torch(self, Xn, yn):
        n=len(Xn); nv=max(1,int(n*self.val_split))
        rng=np.random.default_rng(self.seed); idx=rng.permutation(n)
        vi,ti = idx[:nv], idx[nv:]
        Xtr=torch.tensor(Xn[ti],dtype=torch.float32,device=self._dev)
        ytr=torch.tensor(yn[ti],dtype=torch.float32,device=self._dev)
        Xva=torch.tensor(Xn[vi],dtype=torch.float32,device=self._dev)
        yva=torch.tensor(yn[vi],dtype=torch.float32,device=self._dev)
        dl=DataLoader(TensorDataset(Xtr,ytr),batch_size=max(8,len(ti)//4),shuffle=True)
        self._members=[]
        for k in range(self.n_members):
            torch.manual_seed(self.seed+k)
            m=_TorchMLP(self.d_in,self.hidden,self.dropout).to(self._dev)
            opt=torch.optim.Adam(m.parameters(),lr=self.lr); mse=nn.MSELoss()
            bv,ni=np.inf,0; bs=_copy.deepcopy(m.state_dict())
            for _ in range(2000):
                m.train()
                for xb,yb in dl: opt.zero_grad(); mse(m(xb),yb).backward(); opt.step()
                m.eval()
                with torch.no_grad(): vl=mse(m(Xva),yva).item()
                if vl<bv-1e-6: bv,ni,bs=vl,0,_copy.deepcopy(m.state_dict())
                else:
                    ni+=1
                    if ni>=self.patience: break
            m.load_state_dict(bs); m.eval(); self._members.append(m)

    def _fit_sklearn(self, Xn, yn):
        """sklearn fallback — CI/testing without GPU.
        Disable early_stopping when archive too small for a val split."""
        self._members=[]
        n = len(Xn)
        # Need at least 2 val points; disable early stopping on tiny archives
        use_es = (n >= 10)
        vf = max(self.val_split, 2.0/n) if use_es else 0.1
        for k in range(self.n_members):
            m=_SMLP(hidden_layer_sizes=self.hidden,activation="relu",solver="adam",
                    learning_rate_init=self.lr,max_iter=500,
                    early_stopping=use_es,validation_fraction=vf,
                    n_iter_no_change=self.patience,random_state=self.seed+k,tol=1e-4)
            m.fit(Xn, yn); self._members.append(m)

    def predict(self, x):
        if not self._fitted or not self._members: return 0.0, 1.0
        xn=(np.asarray(x,dtype=np.float64)-self._x_mean)/self._x_std
        if _TORCH and isinstance(self._members[0],_TorchMLP):
            xt=torch.tensor(xn,dtype=torch.float32,device=self._dev).unsqueeze(0)
            with torch.no_grad(): preds=[float(m(xt).item()) for m in self._members]
        else:
            preds=[float(m.predict(xn.reshape(1,-1))[0]) for m in self._members]
        arr=np.array(preds)
        return float(arr.mean())*self._y_std+self._y_mean, max(float(arr.std())*self._y_std,1e-8)

    def mean_abs_error(self, X, y):
        if not self._fitted or len(X)==0: return 1.0
        return float(np.mean([abs(self.predict(xi)[0]-yi) for xi,yi in zip(X,y)]))

_TB = np.array([[64.,256.],[32.,128.],[16.,64.],[0.,0.4],[1e-4,1e-2]])
_TD = len(_TB)

class SurrogateConfig:
    """Chromosome theta=[h1,h2,h3,dropout,lr]; variation via DE in Theta-space."""
    def __init__(self, theta, d_in, seed=0):
        self.theta=theta.copy(); self.d_in=d_in; self.seed=seed; self._model=None

    @classmethod
    def random(cls, d_in, rng, seed=0):
        return cls(rng.uniform(_TB[:,0],_TB[:,1]), d_in, seed)

    def _decode(self):
        return dict(hidden=(int(round(self.theta[0])),int(round(self.theta[1])),int(round(self.theta[2]))),
                    dropout=float(np.clip(self.theta[3],0.,0.4)),
                    lr=float(np.clip(self.theta[4],1e-4,1e-2)))

    def fit(self, X, y):
        p=self._decode()
        self._model=EnsembleMLP(d_in=self.d_in,hidden=p["hidden"],
                                 dropout=p["dropout"],lr=p["lr"],seed=self.seed)
        self._model.fit(X, y)

    def predict(self, x):
        return self._model.predict(x) if self._model else (0.0,1.0)

    def mean_abs_error(self, X, y):
        return self._model.mean_abs_error(X,y) if self._model else 1.0


class SurrogatePopulation:
    """
    M SurrogateConfig individuals.
    Meta-fitness  F_t(theta) = lam*Lpred + (1-lam)*Lopt    Eq.(13).
    Co-evolved via DE/rand/1/bin in Theta-space  (Section 3.2).
    """
    def __init__(self, M, d_in, rng):
        self.M=M; self.d_in=d_in; self.rng=rng
        self.individuals=[SurrogateConfig.random(d_in,rng,seed=i) for i in range(M)]

    def fit_all(self, X, y):
        for c in self.individuals: c.fit(X, y)

    def best(self):
        fitted=[c for c in self.individuals if c._model is not None]
        if not fitted: return self.individuals[0]
        return min(fitted, key=lambda c: c.mean_abs_error([],[]))

    def evolve(self, archive_X, archive_y, solution_pop, lam=0.7, F=0.8, CR=0.9):
        fits=np.array([self._mf(c,archive_X,archive_y,solution_pop,lam) for c in self.individuals])
        new=[]
        for i in range(self.M):
            others=[j for j in range(self.M) if j!=i]
            if len(others) < 3:
                new.append(self.individuals[i]); continue
            r1,r2,r3=self.rng.choice(others,3,replace=False)
            mut=self.individuals[r1].theta+F*(self.individuals[r2].theta-self.individuals[r3].theta)
            mut=np.clip(mut,_TB[:,0],_TB[:,1])
            mask=self.rng.random(_TD)<CR
            if not mask.any(): mask[self.rng.integers(_TD)]=True
            tth=np.where(mask,mut,self.individuals[i].theta)
            t=SurrogateConfig(tth,self.d_in,seed=i); t.fit(archive_X,archive_y)
            tf=self._mf(t,archive_X,archive_y,solution_pop,lam)
            new.append(t if tf<=fits[i] else self.individuals[i])
        self.individuals=new

    @staticmethod
    def _mf(ind, archive_X, archive_y, solution_pop, lam):
        if ind._model is None or len(archive_X)==0: return np.inf
        preds=np.array([ind.predict(x)[0] for x in archive_X])
        Lp=float(np.mean((preds-np.array(archive_y))**2))
        sp=np.array([ind.predict(x)[0] for x in solution_pop])
        Lo=1./(float(np.std(sp))+1e-8)
        return lam*Lp+(1.-lam)*Lo


# ---------------------------------------------------------------------------
# Numpy-only fallback surrogate  (for testing / CI without torch)
# ---------------------------------------------------------------------------

class NumpySurrogate:
    """
    Lightweight RBF surrogate using numpy only.
    Drop-in for EnsembleMLP when torch is not available.
    Returns (y_hat, sigma_hat) where sigma_hat is weighted-std uncertainty.
    """

    def __init__(self, d_in: int, eps: float = 1e-6):
        self.d_in = d_in
        self.eps = eps
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: list, y: list) -> None:
        if len(X) == 0:
            return
        self._X = np.stack(X)
        self._y = np.array(y, dtype=float)
        self._fitted = True

    def predict(self, x: np.ndarray):
        if not self._fitted or self._X is None or len(self._X) == 0:
            return 0.0, 1.0
        dists = np.linalg.norm(self._X - x[None, :], axis=1)
        h = max(float(np.median(dists)), self.eps)
        w = np.exp(-(dists / h) ** 2)
        w_sum = w.sum() + self.eps
        y_hat = float((w @ self._y) / w_sum)
        var = float((w @ (self._y - y_hat) ** 2) / w_sum)
        return y_hat, max(float(np.sqrt(var)), 1e-8)

    def mean_abs_error(self, X: list, y: list) -> float:
        if not self._fitted or len(X) == 0:
            return 1.0
        return float(np.mean([abs(self.predict(xi)[0] - yi)
                               for xi, yi in zip(X, y)]))


class NumpySurrogateConfig:
    """Drop-in replacement for SurrogateConfig without torch."""

    def __init__(self, d_in: int, seed: int = 0):
        self.d_in = d_in
        self.seed = seed
        self.theta = np.array([128.0, 64.0, 32.0, 0.1, 1e-3])
        self._model: Optional[NumpySurrogate] = None
        self._fitness: Optional[float] = None

    def fit(self, X: list, y: list) -> None:
        self._model = NumpySurrogate(self.d_in)
        self._model.fit(X, y)

    def predict(self, x: np.ndarray):
        if self._model is None:
            return 0.0, 1.0
        return self._model.predict(x)

    def mean_abs_error(self, X: list, y: list) -> float:
        if self._model is None:
            return 1.0
        return self._model.mean_abs_error(X, y)


class NumpySurrogatePopulation:
    """Drop-in replacement for SurrogatePopulation without torch."""

    def __init__(self, M: int, d_in: int, rng: np.random.Generator):
        self.M = M
        self.d_in = d_in
        self.rng = rng
        self.individuals = [NumpySurrogateConfig(d_in, seed=i) for i in range(M)]

    def fit_all(self, X: list, y: list) -> None:
        for ind in self.individuals:
            ind.fit(X, y)

    def best(self) -> NumpySurrogateConfig:
        fitted = [ind for ind in self.individuals if ind._model is not None]
        if not fitted:
            return self.individuals[0]
        return min(fitted, key=lambda c: c.mean_abs_error([], []))

    def evolve(self, archive_X, archive_y, solution_pop,
               lam=0.7, F=0.8, CR=0.9) -> None:
        """Simplified: refit all individuals (DE in Theta-space needs torch)."""
        self.fit_all(archive_X, archive_y)
