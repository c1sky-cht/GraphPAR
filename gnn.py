# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv
import tensorlayerx as tlx
from GammaGL.gammagl.layers.conv import GCNConv

# class GNN(nn.Module):
class GNN(tlx.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, activation='leakyrelu'):
        super().__init__()
        self.num_layer = num_layer
        # self.gnns = torch.nn.ModuleList()
        # self.activations = torch.nn.ModuleList()
        self.gnns = tlx.nn.ModuleList()
        self.activations = tlx.nn.ModuleList()
        # self.activations = getattr(tlx.nn, activation)

        for layer in range(self.num_layer):
            if layer == 0:
                gcn = GCNConv(input_dim, hidden_dim)
            else:
                gcn = GCNConv(hidden_dim, hidden_dim)
            self.gnns.append(gcn)

            if activation == 'relu':
                # self.activations.append(nn.ReLU())
                self.activations.append(tlx.nn.activation.ReLU())
            elif activation == 'leakyrelu':
                self.activations.append(tlx.nn.activation.LeakyReLU())
                # self.activations.append(nn.LeakyReLU())
            elif activation == 'prelu':
                self.activations.append(tlx.nn.activation.PReLU())
                # self.activations.append(nn.PReLU())

    #     for gcn in self.gnns:
    #         self.weights_init(gcn)
    #     for act in self.activations:
    #         self.weights_init(act)

    # def weights_init(self, m):
    #     if isinstance(m, Linear):
    #         tlx.nn.initializers.xavier_uniform()(m.weights)
    #         if m.bias is not None:
    #             m.bias.assign(tlx.zeros_like(m.bias))
 

    def forward(self, x, edge_index):
        h = x
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.activations[layer](h)
        return h

    # def __repr__(self):
    #     return (f'{self.__class__.__name__}('
    #             f'num_layer={self.num_layer}, '
    #             f'gnns={self.gnns}, '
    #             f'activations={self.activations})')
    
# class Adapter(torch.nn.Module):
class Adapter(tlx.nn.Module):
    def __init__(self, input_dim, activation):
        super(Adapter, self).__init__()
        self.layer1 = tlx.layers.Linear(in_features = input_dim, out_features = input_dim // 2, W_init='xavier_uniform', b_init=None)
        self.act = tlx.nn.PReLU() if activation == "prelu" else tlx.nn.LeakyReLU()
        self.layer2 = tlx.layers.Linear(in_features = input_dim // 2, out_features = input_dim, W_init='xavier_uniform', b_init=None)
        self.output_layer = tlx.nn.BatchNorm1d(num_features=input_dim, momentum=0.9)
    
    def forward(self, x):
        m = self.layer1(x)
        n = self.act(m)
        z = self.layer2(n)
        return self.output_layer(z)
    
# class Adapter(Module):
#     def __init__(self, input_dim, activation):
#         super(Adapter, self).__init__()
#         self.mlp = tlx.nn.Sequential(Linear(input_dim, input_dim // 2, W_init='xavier_uniform', b_init=None),
#                                     tlx.PReLU() if activation == "prelu" else tlx.nn.LeakyReLU(),
#                                     Linear(input_dim // 2, input_dim, W_init='xavier_uniform', b_init=None))

#         self.output_layer = tlx.nn.BatchNorm1d(num_features=input_dim, momentum=0.9)

#     # def __init__(self, input_dim, activation):
#     #     super(Adapter, self).__init__()
#     #     self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
#     #                              nn.PReLU() if activation == "prelu" else nn.LeakyReLU(),
#     #                              nn.Linear(input_dim // 2, input_dim))

#     #     self.output_layer = nn.BatchNorm1d(input_dim, affine=False)

#     #     for m in self.modules():
#     #         self.weights_init(m)
    
#     def forward(self, x):
#         z = self.mlp(x)
#         return self.output_layer(z)

# class Classifier(nn.Module):
# class Classifier(Module):
#     def __init__(self, input_dim, num_classes):
#         super(Classifier, self).__init__()
#         self.fc1 = Linear(input_dim, num_classes)
#         # self.fc1 = nn.Linear(input_dim, num_classes)
#         self.weights_init(self.fc1)
                
#     def weights_init(self, m):
#         if isinstance(m, Linear):
#             m = tlx.layers.Linear(out_features=num_classes,
#                                         in_features=input_dim,
#                                         W_init='xavier_uniform',
#                                         b_init=None)
#             # tlx.nn.initializers.XavierUniform(m.weight.data)
#             # if m.bias is not None:
#             #     m.bias.data.fill_(0.0)

#     def forward(self, x):
#         ret = self.fc1(x)
#         return ret
class Classifier(tlx.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = tlx.layers.Linear(out_features=num_classes,
                                        in_features=input_dim,
                                        W_init='xavier_uniform',
                                        b_init=None)

    def forward(self, x):
        ret = self.fc1(x)
        return ret
