import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.normal import Normal
import wandb
import seaborn as sns
import pathlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from datetime import date

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_nodes):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        fc1 = nn.Linear(x_dim, hidden_nodes)
        self.encoder = nn.Sequential(fc1, nn.ReLU())
        self.fc21 = nn.Linear(hidden_nodes, z_dim)
        self.fc22 = nn.Linear(hidden_nodes, z_dim)

        fc3 = nn.Linear(z_dim, hidden_nodes)
        self.decoder = nn.Sequential(fc3, nn.ReLU())
        self.fc41 = nn.Linear(hidden_nodes, x_dim)

    def encode(self, x):
        z = self.encoder(x)
        return self.fc21(z), self.fc22(z)

    def decode(self, z):
        x = self.decoder(z)
        return self.fc41(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decode(z)
        return x_mu, z_mu, z_logvar, z

class CSVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, w_dim, hidden_nodes, num_cohorts=None, lik="gaussian"):
        super(CSVAE, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_cohorts = num_cohorts

        if self.num_cohorts is not None:
            input_dim = x_dim + self.num_cohorts
        else:
            input_dim = x_dim

        modules = []
        for d in hidden_nodes:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, d),
                    nn.BatchNorm1d(d, eps=0.001, momentum=0.01),
                    nn.LeakyReLU()
                )
            )
            input_dim = d

        self.encoder = nn.Sequential(*modules)

        # outputs for z
        self.fc21 = nn.Linear(hidden_nodes[-1], z_dim)
        self.fc22 = nn.Linear(hidden_nodes[-1], z_dim)

        # outputs for w
        self.fc21_w = nn.Linear(hidden_nodes[-1], w_dim)
        self.fc22_w = nn.Linear(hidden_nodes[-1], w_dim)

        # decoder
        if self.num_cohorts is not None:
            decoding_dim = z_dim + w_dim + self.num_cohorts
        else:
            decoding_dim = z_dim + w_dim

        modules = []
        for d in hidden_nodes[::-1]:
            modules.append(
                nn.Sequential(
                    nn.Linear(decoding_dim, d),
                    nn.BatchNorm1d(d, eps=0.001, momentum=0.01),
                    nn.LeakyReLU()
                )
            )
            decoding_dim = d

        self.decoder = nn.Sequential(*modules)
        self.fc41 = nn.Linear(hidden_nodes[0], x_dim)

        if "negbin" in lik:
            self.dispersion = nn.Parameter(torch.randn(x_dim), requires_grad=True)
        elif lik != "gaussian":
            assert False, f"invalid likelihood {lik}"

        # linear adversary
        self.fc6 = nn.Linear(z_dim, y_dim)

        # linear decoder of y from w
        self.fc6_w = nn.Linear(w_dim, y_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc21(h), self.fc22(h), self.fc21_w(h), self.fc22_w(h)

    def decode(self, z_w):
        x = self.decoder(z_w)
        return self.fc41(x)

    def decode_y(self, z):
        y = self.fc6(z)
        return y

    def decode_y_from_w(self, w):
        y = self.fc6_w(w)
        return y

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, cohort=None):
        if cohort is not None:
            assert self.num_cohorts is not None
            cohort = F.one_hot(cohort, num_classes=self.num_cohorts) 
            x = torch.cat((x, cohort), 1)
        z_mu, z_logvar, w_mu, w_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        w = self.reparameterize(w_mu, w_logvar)
        z_w = torch.cat((z, w), 1)
        if cohort is not None:
            z_w = torch.cat((z_w, cohort), 1)
        x_mu = self.decode(z_w)
        y_from_w = self.decode_y_from_w(w)
        return x_mu, z_mu, z_logvar, z, w_mu, w_logvar, y_from_w

    def forward_2(self, z, cohort=None):
        y_adv = self.decode_y(z)
        return y_adv
