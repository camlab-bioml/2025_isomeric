import torch
import numpy as np
import scanpy as sc
import scvi
import pandas as pd
import argparse
from datetime import date

def get_cells(train_patients, test_patients, adata, sample_field, cohort): 
    train_cells = []
    for p in train_patients:
        train_cells = train_cells + list(np.where(adata.obs[sample_field] == p)[0])
    train_cells = np.array(train_cells)
    
    test_cells = []
    for p in test_patients:
        test_cells = test_cells + list(np.where(adata.obs[sample_field] == p)[0])
    test_cells = np.array(test_cells)

    if cohort == "peng":
        cell_type_field = "singler.label"
        train_normal_cells = train_cells[np.where( (adata.obs[cell_type_field][train_cells] == "pancreatic ductal cell") | (adata.obs[cell_type_field][train_cells] == "pancreatic acinar cell") )[0]]
        train_tumour_cells = train_cells[np.where(adata.obs[cell_type_field][train_cells] == 'tumour cell')[0]]
        test_normal_cells = test_cells[np.where( (adata.obs[cell_type_field][test_cells] == 'pancreatic ductal cell') | (adata.obs[cell_type_field][test_cells] == 'pancreatic acinar cell') )[0]]
        test_tumour_cells = test_cells[np.where(adata.obs[cell_type_field][test_cells] == 'tumour cell')[0]]
    elif cohort == "zhou":
        train_normal_cells = train_cells[np.where(adata.obs['cell_type'][train_cells] == 'pancreatic epithelial cell (60888)')[0]]
        test_normal_cells = test_cells[np.where(adata.obs['cell_type'][test_cells] == 'pancreatic epithelial cell (60888)')[0]]
        train_tumour_cells = train_cells[np.where(adata.obs['cell_type'][train_cells] == 'tumour cell')[0]]
        test_tumour_cells = test_cells[np.where(adata.obs['cell_type'][test_cells] == 'tumour cell')[0]]
    elif cohort == "spectrum":
        cancer_cell_labels = [i for i in np.unique(adata.obs['cluster_label']) if ('Cancer' in i) | ('cancer' in i)]
        train_tumour_cells = train_cells[[i for i,j in enumerate(adata.obs['cluster_label'][train_cells]) if j in cancer_cell_labels]]
        test_tumour_cells = test_cells[[i for i,j in enumerate(adata.obs['cluster_label'][test_cells]) if j in cancer_cell_labels]]
        # These are the normal ovary HCA dataset cells
        test_normal_cells = np.where(adata.obs["batch"] == "Jin2022")[0]
        train_normal_cells = []
    elif cohort in ["all-lung", "all-breast"]:
        train_tumour_cells = train_cells[np.where(adata.obs['cell_type'][train_cells] == 'Malignant')[0]]
        test_tumour_cells = test_cells[np.where(adata.obs['cell_type'][test_cells] == 'Malignant')[0]]
        train_normal_cells = train_cells[np.where(adata.obs['cell_type'][train_cells] == 'Epithelial')[0]]
        test_normal_cells = test_cells[np.where(adata.obs['cell_type'][test_cells] == 'Epithelial')[0]]
    else:
        print("invalid cohort")
        assert False
    cells = (train_normal_cells, train_tumour_cells, test_normal_cells, test_tumour_cells)
    return cells

def set_dataset_params(cohort):
    path = "/mnt/"
    if cohort == "peng":
        adata = sc.read_h5ad(path + "peng/" + "pdac-peng-tumour-normal-protein-coding-no-ribo-mito.h5ad")
        sample_field = "sample"
        val_patients = ["CRR034513","CRR034519","CRR241802","CRR241799","CRR241800"]

    elif cohort == "zhou":
        adata = sc.read_h5ad(path +"zhou/" + "pdac-zhou-tumour-normal-protein-coding-no-ribo-mito.h5ad")
        sample_field = "case_id"
        val_patients = ["HTA12_10", "HTA12_19", "HTA12_6", "HTA12_12"]
        project_name = "zhou"

    elif cohort == "spectrum":
        adata = sc.read_h5ad(path +"spectrum/" + "spectrum_adata_filtered_tumour_normal-HCA.h5ad")
        sample_field = "batch"
        val_patients = ["SPECTRUM-OV-003", "SPECTRUM-OV-007", "SPECTRUM-OV-025", "SPECTRUM-OV-036",
                        "SPECTRUM-OV-053", "SPECTRUM-OV-067", "SPECTRUM-OV-081", "SPECTRUM-OV-112"]  

    elif cohort == "all-lung":
        adata = sc.read_h5ad(path +"Lung/Lung_adata_filtered_tumour_normal.h5ad")
        sample_field = "sample"
        val_patients = ["NM4E", "P0025", "P0030", "Patient 8", 
                        "RU679", "RU680", "SSN05", "SSN25", "SSN31"]

    elif cohort == "all-breast":
        adata = sc.read_h5ad(path +"Breast/Breast_adata_filtered_tumour_normal.h5ad")
        sample_field = "sample"
        val_patients = ["ATC3", "ER_positive_0360", "ER_positive_LN_0068", "HER2_0331",
                "Patient 43", "TNBC2", "Triple_negative_0126", "ER_positive_LN_0167"]
    else:
        assert False, "Invalid cohort"
    
    all_patients = np.unique(adata.obs[sample_field])

    # handle normal ovary cells
    # by removing the extra cohort
    if (cohort == "spectrum") | (cohort == "all-ovarian"):
        all_patients = all_patients[all_patients != "Jin2022"]

    train_patients = list(set(all_patients).difference(val_patients))
    train_patients = np.sort(train_patients)
    return adata, sample_field, train_patients, val_patients

def main(cohort):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read in tumour+normal adata
    adata, sample_field, train_patients, test_patients = set_dataset_params(cohort)
    cells = get_cells(train_patients, test_patients, adata, sample_field, cohort)
    # these indices index tumour+normal adata
    train_normal_cells, train_tumour_cells, test_normal_cells, test_tumour_cells = cells
    adata_train = adata[train_tumour_cells,:].copy()

    if len(train_normal_cells) == 0:
        assert len(test_normal_cells) > 0
        normal_cells = test_normal_cells
    else:
        normal_cells = np.concatenate((train_normal_cells, test_normal_cells))
    adata_normal = adata[normal_cells,:].copy()

    if cohort in ["peng", "spectrum", "zhou"]:
        scvi.model.SCVI.setup_anndata(adata_train)
    elif cohort in ["all-lung", "all-breast"]:
        scvi.model.SCVI.setup_anndata(adata_train, batch_key = "Cohort")
    else:
        assert False, "invalid cohort"
    model = scvi.model.SCVI(adata_train, n_latent=20)
    model.train(max_epochs=200)
    train_embd = model.get_latent_representation()
    normal_embd = model.get_latent_representation(adata_normal)

    today = str(date.today())
    df = pd.DataFrame(train_embd, index=adata_train.obs.index, columns=[f"z{i}" for i in range(20)])
    df_normal = pd.DataFrame(normal_embd, index=adata_normal.obs.index, columns=[f"z{i}" for i in range(20)])
    df.to_csv(f"{today}-scvi-{cohort}-train-tumour-embd.csv")
    df_normal.to_csv(f"{today}-scvi-{cohort}-normal-embd.csv")

    filepath = f"{today}-scvi-{cohort}" 
    model.save(filepath)
    print("Finished and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train scVI.')
    parser.add_argument('--cohort', type=str, 
                        help='Name of the dataset. Options: peng, zhou, spectrum, all-lung, all-breast')
    args = parser.parse_args()
    main(**vars(args))
