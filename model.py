# -*- coding: utf-8 -*-
# coming soon
import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class My_RGCN(nn.Module):
      def __init__(self,in_size,hidden_size,out_size,num_relations):
          super(My_RGCN,self).__init__()
          self.in_size=in_size
          self.hidden_size=hidden_size
          self.out_size=out_size
          self.num_relations=num_relations

          self.fc1=nn.Linear(self.in_size,self.in_size)
          self.act1=nn.LeakyReLU()
          self.rgcn1=RGCNConv(self.in_size,self.hidden_size,self.num_relations)
          self.rgcn2 = RGCNConv(self.hidden_size, self.hidden_size,self.num_relations)
          self.fc2=nn.Linear(self.hidden_size,self.hidden_size)
          self.out_layer=nn.Linear(self.hidden_size,self.out_size)
          self.act2=nn.LeakyReLU()

      def forward(self,x,edge_index,edge_weight,edge_type):
          x=self.act1(self.fc1(x))
          x=self.rgcn1(x,edge_index,edge_type)
          x = self.rgcn2(x, edge_index, edge_type)
          x=self.act2(self.fc2(x))

          return F.softmax(self.out_layer(x),dim=1)