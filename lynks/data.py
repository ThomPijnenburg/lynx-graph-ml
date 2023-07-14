import torch
import requests
from ogb.nodeproppred import NodePropPredDataset

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_cora(data_root):
    dataset = Planetoid(root=data_root, name='Cora',
                        transform=NormalizeFeatures())
    return dataset


def load_arxiv(data_path):
    # fetch data
    d_name = "ogbn-arxiv"
    dataset = NodePropPredDataset(name=d_name, root=data_path)
    return dataset


def load_arxiv_text(data_path) -> str:
    url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"

    filename = "titleabs.tsv.gz"
    filepath = data_path.joinpath(filename)
    if not filepath.is_file():
        r = requests.get(url, allow_redirects=False)
        open(filepath, "wb").write(r.content)

    return filepath


def rand_edgelist(num_edges=10, num_nodes=100):
    return torch.randint(num_nodes, (2, num_edges))


def sparse_adjacency(edgelist, num_nodes):
    adjacency = torch.sparse_coo_tensor(edgelist,
                                        torch.ones(edgelist.shape[1]),
                                        size=(num_nodes, num_nodes))
    return adjacency


def sparse_identity(length):
    coos = torch.stack((torch.arange(0, length), torch.arange(0, length)))
    identity = torch.sparse_coo_tensor(coos, torch.ones(length))
    return identity


def load_sparse_data(dataset):

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, labels = dataset[0]

    num_nodes = graph["num_nodes"]

    features = torch.tensor(graph["node_feat"])
    labels = torch.LongTensor(labels).squeeze()
    train_idx = torch.LongTensor(train_idx)
    valid_idx = torch.LongTensor(valid_idx)
    test_idx = torch.LongTensor(test_idx)

    # build symmetric adjacency
    adj = sparse_adjacency(graph["edge_index"], num_nodes=num_nodes)
    adj = adj + torch.transpose(adj, 0, 1)
    adj = adj + sparse_identity(num_nodes)

    return adj, features, labels, train_idx, valid_idx, test_idx
