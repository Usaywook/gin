import torch
import torch.nn as nn
import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMMEncoder(nn.Module):
    def __init__(self, num_state, n_iter, verbose=False, tol=1e-6):
        super().__init__()
        self.num_state = num_state
        self.model = GaussianHMM(n_components=self.num_state, n_iter=n_iter, verbose=True, tol=tol)

    def update(self, obs):
        N, C, T, V = obs.shape
        X = obs.permute(0,3,2,1) # (N, V, T, C)
        L = np.array(N * V * [T])
        X = X.reshape(-1, C).detach().cpu().numpy()
        self.model.fit(X, L)

    def forward(self, obs):
        N, C, T, V = obs.shape
        X = obs.permute(0,3,2,1) # (N, V, T, C)
        L = np.array(N * V * [T])
        X = X.reshape(-1, C).detach().cpu().numpy()

        Z = self.model.predict(X, L)

        Z = Z.reshape(N*V, T)[:,-1] # (N*V)
        ans = torch.vstack([torch.eye(self.num_state, device=obs.device)[idx] for idx in Z]) # (N*V, num_state)
        ans = ans.view(N, V, self.num_state)
        return ans

if __name__ == '__main__':
    data = torch.load('./demonstrations/CarlaNavigation/sample.pt')
    state = data['graph_feature']

    model = HMMEncoder(5, 10)
    model.update(state)
    hidden = model(state[160:161])
    print(hidden)
