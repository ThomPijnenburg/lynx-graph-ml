from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_cora(data_root):
    dataset = Planetoid(root=data_root, name='Cora',
                        transform=NormalizeFeatures())
    return dataset
