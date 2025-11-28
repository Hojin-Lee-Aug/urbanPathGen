import sys
import os
import os.path as osp
import shutil
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, EdgeConv, GraphNorm
from torch_geometric.utils import to_undirected
from netCDF4 import Dataset as DatasetNetcdf
import numpy as np
from collections import OrderedDict

BASE_PATH = "./"

class Config:
    uref = 0.0057735
    target_id = '914' # Sample ID
    data_dir = os.path.join("./gcnn/", "data_set")
    checkpoint_path = os.path.join(BASE_PATH, "checkpoint_gcnn.pth.tar")
    output_dir = os.path.join("./gcnn/", "results")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = Config()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


class GCNN(torch.nn.Module):
    def __init__(self):
        super(GCNN, self).__init__()
        var_input = 3; var_output = 2
        self.norm = GraphNorm(var_input)

        # EdgeConv
        self.conv000 = EdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(2*var_input, 28), torch.nn.Tanh(), torch.nn.Linear(28, 28)))
        self.conv00 = EdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(2*28, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64)))
        self.conv01 = EdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(2*64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 128)))
        self.conv02 = EdgeConv(nn=torch.nn.Sequential(torch.nn.Linear(2*128, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256)))

        c_w = 256
        # GraphConv
        self.conv1 = GraphConv(256, c_w); self.conv2 = GraphConv(c_w, 2*c_w)
        self.conv3 = GraphConv(2*c_w, 4*c_w); self.conv4 = GraphConv(4*c_w, 4*c_w)
        self.conv5 = GraphConv(4*c_w, 2*c_w); self.conv6 = GraphConv(2*c_w, c_w)
        self.conv7 = GraphConv(c_w, 128); self.conv8 = GraphConv(128, 64)
        self.conv9 = GraphConv(64, 28); self.conv10 = GraphConv(28, var_output)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        y_0 = self.norm(x, batch)
        y_1 = F.tanh(self.conv000(y_0, edge_index))
        y_2 = F.tanh(self.conv00(y_1, edge_index))
        y_3 = F.tanh(self.conv01(y_2, edge_index))
        y_4 = F.tanh(self.conv02(y_3, edge_index))
        y_5 = F.tanh(self.conv1(y_4, edge_index))
        y_6 = F.tanh(self.conv2(y_5, edge_index))
        y_7 = F.tanh(self.conv3(y_6, edge_index))
        y_8 = F.tanh(self.conv4(y_7, edge_index)) + y_7
        y_9 = F.tanh(self.conv5(y_8, edge_index)) + y_6
        y_10 = F.tanh(self.conv6(y_9, edge_index)) + y_5
        y_11 = F.tanh(self.conv7(y_10, edge_index)) + y_3
        y_12 = F.tanh(self.conv8(y_11, edge_index)) + y_2
        y_13 = F.tanh(self.conv9(y_12, edge_index)) + y_1
        return self.conv10(y_13, edge_index)

def expand_edges_to_2hop(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    A2 = torch.sparse.mm(A, A)
    row, col = A2.coalesce().indices()
    mask = row != col
    edge_combined = torch.cat([edge_index, torch.stack([row[mask], col[mask]])], dim=1)
    return to_undirected(edge_combined).unique(dim=1)

class InferenceDataset(Dataset):
    def __init__(self, root, file_id):
        self.file_id = file_id
        super().__init__(root)
    def len(self): return 1
    def get(self, idx):
        c_path = osp.join(self.root, f"coords/coord_{self.file_id}.hdf5")
        a_path = osp.join(self.root, f"adjLists/adjLst_{self.file_id}.hdf5")
        if not osp.exists(c_path): raise FileNotFoundError(f"no file: {c_path}")

        with h5py.File(c_path, 'r') as f: nodes = f['dataset'][:]
        with h5py.File(a_path, 'r') as f: edges = f['dataset'][:]

        adj = expand_edges_to_2hop(torch.from_numpy(edges.T).long(), nodes.shape[0])
        return Data(x=torch.from_numpy(nodes).float(), edge_index=adj, label=self.file_id)

def main():
    print(f"--- GCNN (ID: {args.target_id}) ---")

    model = GCNN().to(args.device)
    if os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=args.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in ckpt['state_dict'].items() if "pool" not in k}
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ checkpoint load.")
    else:
        print("❌ no ceckpoint file.")
        return

    dataset = InferenceDataset(root=args.data_dir, file_id=args.target_id)
    loader = DataLoader(dataset, batch_size=1)

    model.eval()

    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            pred = model(data)

            save_dir = os.path.join("./gcnn", "results")
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            # PV data
            pv_template_name = f"PV_samplefile.Netcdf"
            pv_template_path = os.path.join(args.data_dir, "samples", pv_template_name)
            output_pv = os.path.join(save_dir, f"PV_pred_{args.target_id}.Netcdf")

            if os.path.exists(pv_template_path):
                shutil.copy(pv_template_path, output_pv) # 복사
                nc = DatasetNetcdf(output_pv, "r+")
                nc.variables['variables0'][:] = pred[:, 0].cpu().numpy()
                nc.variables['variables1'][:] = pred[:, 1].cpu().numpy()
                nc.close()
                print(f"💾 PV data: {output_pv}")
            else:
                print(f"⚠️ No PV template({pv_template_name}). So storing .npy instead of .Netcdf")
                np.save(output_pv.replace(".Netcdf", ".npy"), pred.cpu().numpy())

if __name__ == "__main__":
    main()
