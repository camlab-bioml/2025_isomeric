import torch
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
import wandb
import seaborn as sns
import pathlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from datetime import date
from vae_models import VAE, CSVAE
import umap
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
from torch.nn.functional import gaussian_nll_loss

examples_processed = 0

def loss_function(x_mu, x, z_mu, z_logvar, y_adv, y, w_mu, w_logvar, y_from_w, betas, label_level, lik="gaussian", dispersion=None):

    if lik == "gaussian":
        # Gaussian likelihood
        NLL = F.mse_loss(x_mu, x, reduction="none")

    elif "negbin" in lik:
        assert dispersion is not None
        # NB likelihood
        b, d = x_mu.shape

        if lik == "negbin-sf":
            log_size_factors = torch.log(x.sum(1)) - torch.log(torch.tensor(1e4)) 
            log_size_factors = log_size_factors.unsqueeze(1).expand(-1, x_mu.shape[1])
            x_mu = x_mu + log_size_factors

        ## t_concat is b x d x 2
        t_concat = torch.cat(
                            [x_mu.view(b,d,1), dispersion.view(1,d,1).expand(b,d,1)], 2
                )
        probs = F.softmax(t_concat, dim=2)[:,:,0]
        probs = probs * 0.999
        NLL = -NegativeBinomial(torch.exp(dispersion), probs).log_prob(x)

    else:
        assert False, f"invalid likelihood {lik}"

    # sum over genes and average over batch
    NLL = NLL.sum(1).mean()

    if label_level == "band":
        NEG_ENT = -F.mse_loss(y_adv, y, reduction="none").mean(1)
        W_PREDICTOR = F.mse_loss(y_from_w, y, reduction="none").mean(1)

    elif label_level == "arm":
        NEG_ENT = -F.mse_loss(y_adv, y, reduction="none").sum(1)
        W_PREDICTOR = F.mse_loss(y_from_w, y, reduction="none").sum(1)

    else:
        assert False, f"{label_level} is an unsupported aggregation level for labels"
 
    NEG_ENT = NEG_ENT.mean()
    W_PREDICTOR = W_PREDICTOR.mean()

    KLD, KL = compute_kl(z_mu, z_logvar)
    KLD_w, KL_w = compute_kl(w_mu, w_logvar)
    
    loss1 = betas[0]*NLL + betas[1]*KLD + betas[2]*KLD_w + betas[3]*NEG_ENT + betas[4]*W_PREDICTOR
    unweighted_loss = NLL + KLD + KLD_w + NEG_ENT + W_PREDICTOR

    return loss1, unweighted_loss, NLL, NEG_ENT, KL, KLD, KL_w, KLD_w, W_PREDICTOR

def loss_function_standard(x_mu, x, z_mu, z_logvar, betas):
    NLL = F.mse_loss(x_mu, x, reduction="none")
    NLL = NLL.sum(1).mean()

    KLD, KL = compute_kl(z_mu, z_logvar)

    unweighted_loss = NLL + KLD 
    return betas[0]*NLL + betas[1]*KLD, unweighted_loss, NLL, KL, KLD

def compute_kl(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    KL = KLD.mean(0)
    KLD = KLD.sum(1).mean()
    return KLD, KL

def loss_function_2(y_adv, y, betas, label_level):
    if label_level == "band":
        adv_loss = F.mse_loss(y_adv, y, reduction="none").mean(1)
    elif label_level == "arm":
        adv_loss = F.mse_loss(y_adv, y, reduction="none").sum(1)
    else:
        assert False

    adv_loss = adv_loss.mean()
    return betas[5]*adv_loss

def log_train(model, measures):
    log_dict = {}
    if model.__class__.__name__ == "VAE":
        loss, unweighted_loss, nll, kld, kl = measures

        log_dict["train loss"] = loss.item()
        log_dict["train unweighted loss"] = unweighted_loss.item()
        log_dict["train NLL x"] = nll.item()
        log_dict["train KL z"] = kld.item()
        for d in range(model.z_dim):
            log_dict[f"train KL z dim {d}"] = kl[d].item()

    if model.__class__.__name__ == "CSVAE":
        loss, unweighted_loss, nll, kld, kl, neg_entropy, w_predictor, adv_loss, kld_w, kl_w, norm_grad, max_grad, x_recon_mean = measures

        log_dict["train loss"] = loss.item()
        log_dict["train unweighted loss"] = unweighted_loss.item()
        log_dict["train NLL x"] = nll.item()
        log_dict["train KL z"] = kld.item()
        for d in range(model.z_dim):
            log_dict[f"train KL dim z {d}"] = kl[d].item()
        log_dict["train negative entropy"] = neg_entropy.item()
        log_dict["train w predictor"] = w_predictor.item()
        log_dict["train adversarial loss"] = adv_loss.item()
        log_dict["train KL w"] = kld_w.item()
        for d in range(model.w_dim):
            log_dict[f"train KL w dim {d}"] = kl_w[d].item()

        log_dict["norm grad"] = norm_grad
        log_dict["max grad"] = max_grad
        log_dict["mean(mu_ng)"] = x_recon_mean.item()

    return log_dict

def log_test(model, measures):
    log_dict = {}

    if model.__class__.__name__ == "VAE":
        losses, unweighted_losses, nll_losses, one_kl, kl_losses = measures

        log_dict["test loss"] = np.mean(losses)
        log_dict["test unweighted loss"] = np.mean(unweighted_losses)
        log_dict["test NLL x"] = np.mean(nll_losses)
        log_dict["test KL z"] = np.mean(one_kl)
        for d in range(model.z_dim):
            log_dict[f"test KL z dim {d}"] = kl_losses[d,:].mean()

    if model.__class__.__name__ == "CSVAE":
        losses, unweighted_losses, nll_losses, one_kl, kl_losses, neg_ent_losses, w_predictor_losses, adv_losses, one_kl_w, kl_w_losses = measures

        log_dict["test loss"] = np.mean(losses)
        log_dict["test unweighted loss"] = np.mean(unweighted_losses)
        log_dict["test NLL x"] = np.mean(nll_losses)
        log_dict["test KL z"] = np.mean(one_kl)
        for d in range(model.z_dim):
            log_dict[f"test KL z dim {d}"] = kl_losses[d,:].mean()
        log_dict["test negative entropy"] = np.mean(neg_ent_losses)
        log_dict["test w predictor"] = np.mean(w_predictor_losses)
        log_dict["test adversarial loss"] = np.mean(adv_losses)
        log_dict["test KL w"] = np.mean(one_kl_w)
        for d in range(model.w_dim):
            log_dict[f"test KL w dim {d}"] = kl_w_losses[d,:].mean()

    return log_dict

def train(model, train_loader, optimizer, optimizer2, device, betas, epoch, label_level, lik="gaussian"):
    nlls = []
    global examples_processed
    model.train()

    for batch_idx, data in enumerate(train_loader):
        x = data[0].to(device)

        if model.num_cohorts is not None:
            cohort = data[2].to(device)
        else:
            cohort = None

        if lik == "gaussian":
            dispersion = None
        elif "negbin" in lik:
            dispersion = model.dispersion
        else:
            assert False, f"invalid likelihood {lik}"

        examples_processed = examples_processed + x.shape[0]
        if model.__class__.__name__ != "VAE":
            y = data[1].to(device)

        optimizer.zero_grad()

        if model.__class__.__name__ == "VAE":
            x_recon, z_mu, z_logvar, z = model.forward(x)
        elif model.__class__.__name__ == "CSVAE":
            x_recon, z_mu, z_logvar, z, w_mu, w_logvar, y_from_w = model.forward(x, cohort)
        else: 
            print("Unsupported model")
            assert False

        if model.__class__.__name__ == "CSVAE":
            y_logprobs = model.forward_2(z, cohort)
            loss, unweighted_loss, nll, neg_entropy, kl, kld, kl_w, kld_w, w_predictor = loss_function(x_recon, x, z_mu, z_logvar, y_logprobs, y,
                                                                                    w_mu, w_logvar, y_from_w, 
                                                                                    betas, label_level, lik, dispersion)
        else:
            loss, unweighted_loss, nll, kl, kld = loss_function_standard(x_recon, x, z_mu, z_logvar, betas)

        loss.backward()
        norm_grad = np.sqrt(sum([(torch.norm(i.grad)**2).item() for i in model.parameters()]))
        max_grad = np.max([torch.max(torch.abs(i.grad)).item() for i in model.parameters()])
        optimizer.step()
        nlls.append(nll.item())

        if model.__class__.__name__ == "CSVAE":
            optimizer2.zero_grad()
            y_logprobs2 = model.forward_2(z.detach(), cohort)
            adv_loss = loss_function_2(y_logprobs2, y, betas, label_level)
            adv_loss.backward()
            optimizer2.step()

        if model.__class__.__name__ == "VAE":
            measures = (loss, unweighted_loss, nll, kld, kl)
        elif model.__class__.__name__ == "CSVAE":
            measures = (loss, unweighted_loss, nll, kld, kl, neg_entropy, w_predictor, adv_loss, kld_w, kl_w, norm_grad, max_grad, torch.exp(x_recon).mean())

        log_dict = log_train(model, measures)
        log_dict["examples processed"] = examples_processed
        log_dict["epoch"] = epoch
        wandb.log(log_dict)

    return np.mean(nlls)

def test(model, test_loader, device, betas, epoch, label_level, lik="gaussian"):
    losses = []
    unweighted_losses = []
    nll_losses = []
    nll_y_losses = []
    neg_ent_losses = []
    w_predictor_losses = []
    if model.__class__.__name__ in ["CSVAE", "VAE"]:
        kl_losses = np.zeros((model.z_dim, len(test_loader))) 
    if model.__class__.__name__ == "CSVAE":
        kl_w_losses = np.zeros((model.w_dim, len(test_loader))) 
    one_kl = []
    one_kl_w = []
    adv_losses = []
    log_dict = {}

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data[0].to(device)

            if model.num_cohorts is not None:
                cohort = data[2].to(device)
            else:
                cohort = None

            if lik == "gaussian":
                dispersion = None
            elif "negbin" in lik:
                dispersion = model.dispersion
            else:
                assert False, f"invalid likelihood {lik}"

            if model.__class__.__name__ != "VAE":
                y = data[1].to(device)

            if model.__class__.__name__ == "VAE":
                x_recon, z_mu, z_logvar, z = model.forward(x)
            elif model.__class__.__name__ == "CSVAE":
                x_recon, z_mu, z_logvar, z, w_mu, w_logvar, y_from_w = model.forward(x, cohort)
            else: 
                print("Unsupported model")
                assert False

            if model.__class__.__name__ == "CSVAE":
                y_logprobs = model.forward_2(z, cohort)
                loss, unweighted_loss, nll, neg_entropy, kl, kld, kl_w, kld_w, w_predictor = loss_function(x_recon, x, z_mu, z_logvar, y_logprobs, y,
                                                                                        w_mu, w_logvar, y_from_w, 
                                                                                        betas, label_level, lik, dispersion)
                y_logprobs2 = model.forward_2(z.detach(), cohort)
                adv_loss = loss_function_2(y_logprobs2, y, betas, label_level)

            else:
                loss, unweighted_loss, nll, kl, kld = loss_function_standard(x_recon, x, z_mu, z_logvar, betas)

            losses.append(loss.item())
            unweighted_losses.append(unweighted_loss.item())
            nll_losses.append(nll.item())

            one_kl.append(kld.item())
            for d in range(model.z_dim):
                kl_losses[d,i] = kl[d].item()

                if model.__class__.__name__ == "CSVAE":
                    neg_ent_losses.append(neg_entropy.item())
                    w_predictor_losses.append(w_predictor.item())
                    adv_losses.append(adv_loss.item())
                    one_kl_w.append(kld_w.item())
                    for d in range(model.w_dim):
                        kl_w_losses[d,i] = kl_w[d].item()

        if model.__class__.__name__ == "VAE":
            measures = (losses, unweighted_losses, nll_losses, one_kl, kl_losses)

        if model.__class__.__name__ == "CSVAE":
            measures = (losses, unweighted_losses, nll_losses, one_kl, kl_losses, neg_ent_losses, w_predictor_losses, adv_losses, one_kl_w, kl_w_losses)

        log_dict = log_test(model, measures)
        log_dict["examples processed"] = examples_processed
        log_dict["epoch"] = epoch
        wandb.log(log_dict)
        return np.mean(losses), np.mean(unweighted_losses)

def train_val_split_by_patient(train_patients, val_patients, data, sample_field, labels, batch_size, cohort, num_cohorts=None, lik="gaussian"):
    train_indices = (data.obs[sample_field].isin(train_patients)).tolist()
    val_indices = (data.obs[sample_field].isin(val_patients)).tolist()
    train_indices = [i for i, x in enumerate(train_indices) if x]
    val_indices = [i for i, x in enumerate(val_indices) if x]

    all_train_data = data[train_indices,:]
    all_val_data = data[val_indices,:]

    train_data = all_train_data
    val_data = all_val_data

    if num_cohorts is not None:
        enc = LabelEncoder()
        train_cohorts = enc.fit_transform(train_data.obs["Cohort"])
        val_cohorts = enc.fit_transform(val_data.obs["Cohort"])
        train_cohorts = torch.tensor(train_cohorts, dtype=torch.long)
        val_cohorts = torch.tensor(val_cohorts, dtype=torch.long)
    
    # normalize
    train_data = normalize_sc_data(train_data, lik)
    val_data = normalize_sc_data(val_data, lik)
    
    # cast to torch
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)

    if len(labels) > 0:
        # split labels
        all_train_labels = labels[train_indices,:]
        all_val_labels = labels[val_indices,:]

        train_labels = all_train_labels
        val_labels = all_val_labels
        
        # cast to torch
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.float32)
        
        if num_cohorts is not None:
            train_ds = TensorDataset(train_data, train_labels, train_cohorts)
            val_ds = TensorDataset(val_data, val_labels, val_cohorts)

        else:
            train_ds = TensorDataset(train_data, train_labels)
            val_ds = TensorDataset(val_data, val_labels)

    else:
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)

        if cohort in ["all_lung"]:
            assert False, "Cohort indicators are not implemented when there are no labels."

    # make dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False
    )
    
    outputs = (train_loader, val_loader, np.array(train_indices), np.array(val_indices))
    return outputs

def configure_params(model, lik="gaussian"):
    if model.__class__.__name__ == "CSVAE":
        l1_params = list(model.encoder.parameters()) \
                    + list(model.fc21.parameters()) \
                    + list(model.fc22.parameters()) \
                    + list(model.decoder.parameters()) \
                    + list(model.fc41.parameters()) \
                    + list(model.fc21_w.parameters()) \
                    + list(model.fc22_w.parameters()) \
                    + list(model.fc6_w.parameters()) 

        if "negbin" in lik:
            l1_params = l1_params + [model.dispersion]
        elif lik != "gaussian":
            assert False, f"invalid likelihood {lik}"

        l2_params = list(model.fc6.parameters())

    else:
        l1_params = model.parameters()
        l2_params = []

    return l1_params, l2_params

def get_data_tensor(adata, lik="gaussian"):
    with torch.no_grad():
        adata_norm = normalize_sc_data(adata.copy(), lik)
        data_tensor = torch.tensor(adata_norm, dtype=torch.float32)
    return data_tensor

def normalize_sc_data(adata, lik="gaussian", target_sum=1e4):
    if lik == "gaussian":
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    if lik not in ["gaussian", "negbin", "negbin-sf"]:
        assert False, f"invalid likelihood {lik}"
    return np.array(adata.X.todense())

def get_subsets(values, indices):
    train_id, test_id = indices

    train_values = values[train_id,:]    
    test_values = values[test_id,:]    

    values = (train_values, test_values)
    return values  

def set_dataset_params(cohort, label_level):
    path = "/ddn_exa/campbell/share/datasets/ISOMERIC_2024/data-cna-effects-project/"
    if cohort == "peng":
        adata = sc.read_h5ad(path + "peng/" + "pdac-peng-tumour-cells-protein-coding-no-ribo-mito.h5ad")
        
        if label_level == "band":
            labels = pd.read_csv(path + "peng/" + "pdac-peng-tumour-cells-band-labels.csv", index_col=0)
        elif label_level == "arm":
            labels = pd.read_csv(path + "peng/" + "pdac-peng-tumour-cells-arm-labels.csv", index_col=0)
        else:
            assert False

        sample_field = "sample"
        val_patients = ["CRR034513","CRR034519","CRR241802","CRR241799","CRR241800"]
        project_name = "peng-tumour"

    elif cohort == "zhou":
        adata = sc.read_h5ad(path + "zhou/" + "pdac-zhou-tumour-cells-protein-coding-no-ribo-mito.h5ad")
        
        if label_level == "band":
            labels = pd.read_csv(path + "zhou/" + "pdac-zhou-tumour-cells-band-labels.csv", index_col=0)
        elif label_level == "arm":
            labels = pd.read_csv(path + "zhou/" + "pdac-zhou-tumour-cells-arm-labels.csv", index_col=0)
        else:
            assert False

        sample_field = "case_id"
        val_patients = ["HTA12_10", "HTA12_19", "HTA12_6", "HTA12_12"]
        project_name = "zhou"

    elif cohort == "spectrum":
        adata = sc.read_h5ad(path + "spectrum/" + "spectrum-matched-labels-train-val-no-ribo-mito-chrom-annotated.h5ad")

        if label_level == "band":
            labels = pd.read_csv(path + "spectrum/" + "spectrum-cont-labels-3-classes-train-val-tumour-bands.csv", index_col=0)
        elif label_level == "arm":
            labels = pd.read_csv(path + "spectrum/" + "spectrum-cont-labels-3-classes-train-val-tumour.csv", index_col=0)
        else:
            assert False

        sample_field = "batch"
        val_patients = ["SPECTRUM-OV-003", "SPECTRUM-OV-007", "SPECTRUM-OV-025", "SPECTRUM-OV-036",
                        "SPECTRUM-OV-053", "SPECTRUM-OV-067", "SPECTRUM-OV-081", "SPECTRUM-OV-112"]  
        project_name = "spectrum_ov"

    elif cohort == "all-lung":
        adata = sc.read_h5ad("/ddn_exa/campbell/share/datasets/3CA-datasets/Lung/Lung_adata_filtered_tumour_only_nonzero_genes_train_val_no_maynard.h5ad")

        if label_level == "band":
            labels = pd.read_csv("/ddn_exa/campbell/share/datasets/3CA-datasets/Lung/LungCNA_labels_tumour_only_train_val_no_maynard_bands.csv", index_col=0)
        elif label_level == "arm":
            labels = pd.read_csv("/ddn_exa/campbell/share/datasets/3CA-datasets/Lung/LungCNA_labels_tumour_only_train_val_no_maynard_tf.csv", index_col=0)
        else:
            assert False

        sample_field = "sample"
        val_patients = ["NM4E", "P0025", "P0030", "Patient 8", 
                        "RU679", "RU680", "SSN05", "SSN25", "SSN31"]
        project_name = "lung"

    elif cohort == "all-breast":
        adata = sc.read_h5ad("/ddn_exa/campbell/share/datasets/3CA-datasets/Breast/Breast_adata_filtered_tumour_only_nonzero_genes_train_val.h5ad")

        if label_level == "band":
            labels = pd.read_csv("/ddn_exa/campbell/share/datasets/3CA-datasets/Breast/BreastCNA_labels_tumour_only_train_val_bands.csv", index_col=0)
        elif label_level == "arm":
            labels = pd.read_csv("/ddn_exa/campbell/share/datasets/3CA-datasets/Breast/BreastCNA_labels_tumour_only_train_val_tf.csv", index_col=0)
        else:
            assert False

        sample_field = "sample"
        val_patients = ["ATC3", "ER_positive_0360", "ER_positive_LN_0068", "HER2_0331",
                "Patient 43", "TNBC2", "Triple_negative_0126", "ER_positive_LN_0167"]
        project_name = "breast"

    else:
        assert False, "Invalid cohort"
    
    all_patients = np.unique(adata.obs[sample_field])
    train_patients = list(set(all_patients).difference(val_patients))
    train_patients = np.sort(train_patients)
    label_names = labels.columns.tolist()
    labels = np.array(labels)

    return adata, labels, label_names, sample_field, train_patients, val_patients, project_name

def main(cohort, label_level, lik, cohort_encoding, scheduler, n_layers, negent_weight, lr, seed, epochs):
    if cohort_encoding == "cohort":
        cohort_encoding = True
    else:
        cohort_encoding = False

    if scheduler == "scheduler":
        scheduler = True
    else:
        scheduler = False
    
    m = "CSVAE"
    adata, labels, label_names, sample_field, train_patients, val_patients, project_name = set_dataset_params(cohort, label_level)

    if m == "VAE": 
        y_dim = 0
    else:
        y_dim = labels.shape[1]

    x_dim = adata.shape[1]

    if cohort_encoding:
        num_cohorts = np.unique(adata.obs["Cohort"]).shape[0]
        enc = LabelEncoder()
        cohorts = enc.fit_transform(adata.obs["Cohort"])
    else:
        num_cohorts = None
        cohorts = None

    w_dim = 10
    z_dim = 10
    hidden_nodes = np.repeat(128, n_layers).tolist()

    if label_level == "band":
        assert negent_weight == 1, "can't set negent_weight when using bands-level model"
        betas = [1, 1, 1, 40, 40, 40]
    elif label_level == "arm":
        betas = [1, 1, 1, negent_weight, 1, 1]
    else:
        assert False, "unsupported label aggregation level"

    batch_size = 256
    
    outputs = train_val_split_by_patient(train_patients, val_patients, adata, sample_field, labels, batch_size, cohort, num_cohorts, lik)

    train_loader, test_loader, train_indices, val_indices = outputs 
    indices = (train_indices, val_indices) 

    config = dict(model=m,
                cohort=cohort,
                label_level=label_level,
                lik=lik,
                cohort_encoding=cohort_encoding,
                x_dim=x_dim,
                y_dim=y_dim,
                epochs=epochs,
                seed=seed,
                batch_size=batch_size,
                betas=betas,
                n_layers=n_layers,
                negent_weight=negent_weight,
                lr=lr)
    wandb.init(project=project_name, config=config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if m == "VAE":
        model = VAE(x_dim, z_dim, hidden_nodes).to(device)
    elif m == "CSVAE":
        model = CSVAE(x_dim, y_dim, z_dim, w_dim, hidden_nodes, num_cohorts, lik).to(device)
    else:
        print(f"{m} is not a supported model")
        assert False

    l1_params, l2_params = configure_params(model, lik)

    optimizer = optim.Adam(l1_params, lr=lr)
    if len(l2_params) > 0:
        optimizer2 = optim.Adam(l2_params, lr=lr)
    else:
        optimizer2 = [] 
    if scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=0.001)

    nans_present = False
    for epoch in range(epochs):
        val_loss, val_unweighted_loss = test(model, test_loader, device, betas, epoch, label_level, lik)
        nll_epoch = train(model, train_loader, optimizer, optimizer2, device, betas, epoch, label_level, lik)
        if np.isnan(nll_epoch):
            nans_present = True
            break
        if scheduler:
            scheduler.step(nll_epoch)

    print("Finished training.")
    if nans_present:
        print("NaN train loss.")

    wandb.finish()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train ISOMERIC on scRNA-seq and CNA calls.')

    parser.add_argument('--cohort', type=str, 
                        help='Name of the dataset. Options: peng, zhou, spectrum, all-lung, all-breast')

    parser.add_argument('--label_level', type=str,
                        help='Level of CNA labels. Options: arm, band')

    parser.add_argument('--lik', type=str, default='gaussian',
                        help='Likelihood. Options: gaussian, negbin, negbin-sf')

    parser.add_argument('--cohort_encoding', type=str, default='no cohort',
                        help='Add cohort encoding to the model? Set "cohort" for pooled datasets.')

    parser.add_argument('--scheduler', type=str, default='no scheduler', help='Use scheduler or not.')

    parser.add_argument('--n_layers', type=int, default=1, help='Number of hidden layers.')

    parser.add_argument('--negent_weight', type=float, default=1., help='Weight for the adversarial term.')

    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')

    parser.add_argument('--seed', type=int, default=128, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')

    args = parser.parse_args()
    main(**vars(args))
