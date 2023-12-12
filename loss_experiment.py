#!/usr/bin/env python3

from adverserial_MI_model import main as run_model
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from anndata import read_h5ad
from torch import save as torch_save
from collections import defaultdict
import csv
import argparse
import os


def main(embeddings_path, run_name, 
    save_model = True,
    cv = 5,
    split_type = 'group', 
    domain_col = 'gname', 
    reg = 1.,
):

    params = {
            'batch_size': 64,
            'num_epochs': 50,
            'reg_lambda': reg,
            'temperature_optim_lr': 0.00030582134656912167,
            'discriminator_optim_lr': 0.0112742004919829,
            'dropout': 0.013221096935269972,
            'gamma': 0.8122225720036048,
            'hidden_size': 64
        }
    
    embeddings = read_h5ad(embeddings_path)
    
    labelencoder = LabelEncoder().fit( embeddings.obs[domain_col].values[:,None] )

    if split_type == 'group':
        splitter = GroupShuffleSplit(
            n_splits=cv, test_size=0.2, random_state=0
        ).split(
            X = embeddings,
            y = embeddings.obs.optimum_tmp,
            groups = embeddings.obs[domain_col].astype(str).fillna('none').values
        )
    elif split_type == 'random':
        splitter = ShuffleSplit(
            n_splits=cv, test_size=0.2, random_state=0
        ).split(
            X = embeddings,
            y = embeddings.obs.optimum_tmp,
        )
    else:
        raise ValueError()
    

    test_losses = defaultdict(list); train_losses = defaultdict(list)

    for rep, (train_idx, test_idx) in enumerate(splitter):        
        
        scaler = StandardScaler().fit( embeddings[train_idx].obs.optimum_tmp.values[:,None] )
        
        models, test_loss, train_loss = run_model(
            train_adata=embeddings[train_idx].copy(),
            test_adata=embeddings[test_idx].copy(),
            temperature_transformer=scaler,
            homolog_transformer=labelencoder,
            homolog_column=domain_col,
            temperature_column='optimum_tmp',
            eval_every=5,
            log_name=f'runs/{run_name}/{rep}',
            **params,
        )

        if rep == 0 and save_model:
            if not os.path.isdir('models'): os.mkdir('models/')
            torch_save(models, f'models/{run_name}.models.pt')

        for key, value in test_loss.items():
            test_losses[key].append(value)

        for key, value in train_loss.items():
            train_losses[key].append(value)
    

    # Write test_losses and train_losses to TSV file
    if not os.path.isdir('results'): os.mkdir('results')

    with open(f'results/{run_name}.losses.tsv', 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['Metric', 'Rep','Test Loss', 'Train Loss'])
        for key in test_losses:
            for rep in range(cv):
                writer.writerow([key, rep, test_losses[key][rep], train_losses[key][rep]])

    
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_path', type = str)
    parser.add_argument('run_name', type = str)
    parser.add_argument('--cv-iters','-cv', type = int, default=5)
    parser.add_argument('--split-type', type = str, choices = ['group','random'], default = 'group')
    parser.add_argument('--domain-col', type = str, default = 'gname')
    parser.add_argument('--reg', type = float, default = 1.)

    args = parser.parse_args()

    main(
        embeddings_path = args.embeddings_path,
        run_name = args.run_name,
        save_model=True,
        cv = args.cv_iters,
        split_type = args.split_type,
        domain_col = args.domain_col,
        reg = args.reg,
    )