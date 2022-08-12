from matplotlib.pyplot import axis
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TPFeeder(Dataset):
    def __init__(self, data_path, train=True, test=False, weighted=False):
        super().__init__()
        self.data = torch.load(data_path, map_location=torch.device('cpu'))

        total_num = len(self.data['graph_feature'])
        if test:
            id_list = list(range(total_num))
        else:
            train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
            val_id_list = list(set(list(range(total_num))) - set(train_id_list))
            id_list = train_id_list if train else val_id_list

        self.nodes = self.data['graph_feature'][id_list]
        self.adjs = self.data['weighted_adjacency'][id_list] if weighted else self.data['adjacency'][id_list]

        self.num_frame = int(self.nodes.shape[2] / 2)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx, :, :self.num_frame, :], self.adjs[idx], self.nodes[idx, :, self.num_frame:, :]

if __name__=='__main__':
    loader = DataLoader(TPFeeder(data_path='demonstrations/CarlaNavigation/rule.pt'), batch_size=4, shuffle=True, drop_last=True)

    nodes, adjs, gts = iter(loader).next()

    import matplotlib.pyplot as plt
    for nd, ad, gnd in zip(nodes, adjs, gts):
        nd = np.transpose(nd.cpu().numpy(), (2,1,0))
        gnd = np.transpose(gnd.cpu().numpy(), (2,1,0))
        num_obj = np.sum(nd[:,-1,-1]).astype(int)
        for past, future in zip(nd[:num_obj,:,[0,1,-1]], gnd[:num_obj,:,[0,1,-1]]):
            past = past[past[:,-1]==1]
            future = future[future[:,-1]==1]

            plt.scatter(past[:,0], past[:,1], c='k', alpha=0.2)
            plt.scatter(past[-1,0], past[-1,1], c='cyan', alpha=1.0)
            plt.scatter(future[:,0], future[:,1], c='r', alpha=0.2)
        plt.xlim(-45,45)
        plt.ylim(-45,45)
        plt.title(f'num : {num_obj}')
        plt.show()
        plt.cla()