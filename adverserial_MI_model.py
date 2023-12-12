import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from anndata import read_h5ad
from numpy import newaxis
import argparse
import joblib
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
import numpy as np
from tqdm import trange


class AnnDataDataset(Dataset):

    def __init__(self, adata,*,
                    n_classes,
                    temperature_column = 'optimal_temperature', 
                    ph_column = 'optimal_pH',
                    homolog_column = 'gene_family',
                    device = 'cpu',
                    temperature_transform=lambda x : x, 
                    homolog_transform=lambda x : x,
                    
        ):

        self.n_classes = n_classes
        self.data = adata
        self.temperature_transform = temperature_transform
        self.homolog_transform = homolog_transform
        self.temperature = self.temperature_transform( self.data.obs[temperature_column].values[:,newaxis] )
        self.homolog = self.homolog_transform( self.data.obs[homolog_column].values )
        self.device = device
        #self.ph = self.data.obs[ph_column].values

    def __len__(self):
        return len(self.temperature)

    def __getitem__(self, idx):
        
        return (
            torch.tensor(self.data.X[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.temperature[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.homolog[idx], dtype=torch.long, device=self.device),
        )
    
    @property
    def n_dims(self):
        return self.data.X.shape[1]
    


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x):
        res = x
        x = self.block(x)
        x += res
        return x
    

def get_fc_network(*,input_dim, output_dim, hidden_dim, n_layers,
                   dropout = 0.1):

    def middle_layer(input_dim, output_dim):
        return nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )

    layers = [middle_layer(input_dim, hidden_dim)]
    
    # Add hidden layers
    for _ in range(n_layers-2):
        layers.append(ResidualAdd(middle_layer(hidden_dim, hidden_dim)))
    
    # Add output layer
    layers.append(nn.Sequential(
        nn.Linear(hidden_dim, output_dim),
    ))
    
    return nn.Sequential(*layers)


def forward(
    sequence_features, temperature, homolog_class,*,
    representation_model,
    temperature_model,
    discriminator_model,
    reg_lambda,
):
    
    # Compute the representation
    x = representation_model(sequence_features)
    
    # Compute the temperature loss
    temperature_loss = F.mse_loss(temperature_model(x), temperature)
    # Compute the adverserial loss
    y_hat = discriminator_model(x)
    # Compute the total loss
    loss = temperature_loss - reg_lambda*F.cross_entropy(y_hat, homolog_class)

    # Compute the discriminator loss
    discriminator_loss = F.cross_entropy(
                            discriminator_model(x.detach()),
                            homolog_class
                         )

    return {
        'adverserial_loss' : loss,
        'temperature_loss' : temperature_loss,
        'discriminator_loss' : discriminator_loss,
    }


def backward(
    adverserial_loss,
    temperature_loss,
    discriminator_loss,
    temperature_optim,
    discriminator_optim,
):

    temperature_optim.zero_grad()
    adverserial_loss.backward()
    temperature_optim.step()

    discriminator_optim.zero_grad()
    discriminator_loss.backward()
    discriminator_optim.step()
    
    

def fit(*,
    dataset,
    models,
    writer,
    reg_lambda,
    num_epochs = 500,
    batch_size = 128,
    temperature_optim_lr = 5e-3,
    discriminator_optim_lr = 5e-3,
    gamma = 0.9,
    ):

    optimizers = {
        'temperature_optim' : torch.optim.Adam(
                                    [
                                        *models['representation_model'].parameters(), 
                                        *models['temperature_model'].parameters()
                                    ], lr = temperature_optim_lr,
                                ),
        'discriminator_optim' : torch.optim.Adam(
                                    models['discriminator_model'].parameters(),
                                    lr = discriminator_optim_lr,
                                ),
    }

    batches_per_epoch = len(dataset)//batch_size

    scheduler1 = torch.optim.lr_scheduler.StepLR(
        optimizers['temperature_optim'],
        step_size = batches_per_epoch,
        gamma = gamma,
    )

    scheduler2 = torch.optim.lr_scheduler.StepLR(
        optimizers['discriminator_optim'],
        step_size = batches_per_epoch,
        gamma = gamma,
    )

    adverserial_penalty_annealer = lambda step_num : 1.

    # Get the data loader
    train_set = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Train the model
    step_num = 0
    for epoch in trange(num_epochs, desc = 'Epoch', ncols = 100, leave = True):

        writer.add_scalar('LR/temperature', scheduler1.get_last_lr()[0], epoch)
        writer.add_scalar('LR/discriminator', scheduler2.get_last_lr()[0], epoch)
        
        for m in models.values():
            m.train()

        train_losses = defaultdict(lambda : 0.)
        # Iterate over the data
        for batch in train_set:

            # Forward and backward pass
            losses = forward(*batch, **models, reg_lambda= adverserial_penalty_annealer(step_num)*reg_lambda)
            backward(**losses, **optimizers)
            
            scheduler1.step(); scheduler2.step()
            step_num+=1

            for l, v in losses.items():
                train_losses[l] += v

        for l,v in train_losses.items():
            writer.add_scalar(f'Train/{l}', v/batches_per_epoch, epoch)

        writer.add_scalar('Train/lambda', adverserial_penalty_annealer(step_num)*reg_lambda, epoch)

        yield models


def score(*,
    dataset,
    models,
    reg_lambda,
):
    
    for m in models.values():
        m.eval()

    # Get the data loader
    test_set = DataLoader(dataset, 
                          batch_size=len(dataset), 
                          shuffle=False, 
                          drop_last=False
                        )

    batch = next(iter(test_set))
    losses = forward(*batch, **models, reg_lambda=reg_lambda)
    return {k : v.item() for k,v in losses.items()}



def predict(*,dataset, models):

    for m in models.values():
        m.eval()

    # Get the data loader
    dataset = DataLoader(dataset, 
                          batch_size=len(dataset), 
                          shuffle=False, 
                          drop_last=False
                        )

    sequence_features, temperature, homolog_class = next(iter(dataset))
    # Compute the representation
    return {
        'embedding' : ( models['representation_model'](sequence_features) ).detach().cpu().numpy(),
        'temperature' : temperature.detach().cpu().numpy(),
        'homolog_class' : homolog_class.detach().cpu().numpy(),
    }


def main(
    *,
    train_adata,
    test_adata,
    temperature_transformer,
    homolog_transformer,
    temperature_column = 'optimal_temperature', 
    ph_column = 'optimal_pH',
    homolog_column = 'gene_family',
    device = 'cpu',
    n_layers = 3,
    reg_lambda = 1.,
    num_epochs = 500,
    batch_size = 128,
    temperature_optim_lr = 5e-3,
    discriminator_optim_lr = 5e-3,
    seed = 0,
    eval_every = 10,
    dropout = 0.05,
    callback = None,
    save_embedding = False,
    hidden_size = None,
    log_name = None,
    gamma = 0.9,
    ):

    # Set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create the dataset
    dataset_kw = dict(
        temperature_column = temperature_column, 
        ph_column = ph_column,
        homolog_column = homolog_column,
        device = device,
        temperature_transform = temperature_transformer.transform,
        homolog_transform = homolog_transformer.transform,
        n_classes = len(homolog_transformer.classes_)
    )

    train_dataset = AnnDataDataset(train_adata, **dataset_kw)
    test_dataset = AnnDataDataset(test_adata, **dataset_kw)
    examples = Subset(train_dataset, np.random.choice(len(train_dataset), 5000, replace=False))

    dim = train_dataset.n_dims
    num_homologs = train_dataset.n_classes
    hidden_size = hidden_size or dim

    # Create the models
    models = {
        'representation_model' : get_fc_network(
            input_dim = dim, 
            output_dim = hidden_size, 
            hidden_dim = hidden_size, 
            n_layers = n_layers,
            dropout=dropout,
        ).to(device),
        'temperature_model' : nn.Linear(hidden_size, 1).to(device),
        'discriminator_model' : nn.Linear(hidden_size, num_homologs).to(device),
    }

    writer = SummaryWriter(log_dir=log_name)
    
    try:
        # Train the model
        for epoch, models in enumerate(
            fit(
                dataset = train_dataset,
                models = models,
                writer = writer,
                reg_lambda = reg_lambda,
                num_epochs = num_epochs,
                batch_size = batch_size,
                temperature_optim_lr = temperature_optim_lr,
                discriminator_optim_lr = discriminator_optim_lr,
                gamma = gamma,
            )
        ):
            if epoch % eval_every == 0 and epoch > 0:

                if save_embedding and epoch == num_epochs-1:
                    
                    embedding_space = predict(dataset = examples, models = models)

                    writer.add_embedding(
                        embedding_space['embedding'], 
                        metadata = embedding_space['homolog_class'],
                        global_step = epoch,
                        tag = 'Embedding',
                    )

                test_losses = score(
                            dataset = test_dataset, 
                            models = models, 
                            reg_lambda = reg_lambda
                        )
                
                for l,v in test_losses.items():
                    writer.add_scalar(f'Test/{l}', v, epoch)

                if not callback is None:
                    callback(epoch = epoch,**models, **test_losses)

    except KeyboardInterrupt:
        logger.info('Interrupted')
        pass
        
    test_loss = score(
            dataset = test_dataset, 
            models = models, 
            reg_lambda = reg_lambda
        )

    train_loss = score(
            dataset = train_dataset, 
            models = models, 
            reg_lambda = reg_lambda
        )
    
    return models, test_loss, train_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the adverserial model")

    def valid_path(path):
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"Invalid path {path}")
        return path
    
    # Add command line arguments
    parser.add_argument('--output', '-o', type=str, required=True, help="Output path")
    parser.add_argument("--train-anndata-path", '-train', type=valid_path, help="Path to the train anndata file")
    parser.add_argument("--test-anndata-path", '-test', type=valid_path, help="Path to the test anndata file")
    parser.add_argument("--temperature-transformer", '-tt', type=valid_path, help="Path to the temperature transformer")
    parser.add_argument("--homolog-transformer", '-ht', type=valid_path, help="Path to the homolog transformer")
    parser.add_argument("--temperature-column", '-temp-col', type=str, default="optimal_tmp", help="Temperature column name")
    parser.add_argument("--ph-column", '-pH-col', type=str, default="optimal_pH", help="pH column name")
    parser.add_argument("--homolog-column", '-homolog-col', type=str, default="gene_family", help="Homolog column name")
    parser.add_argument("--device", '-d', type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--n-layers", '-n',type=int, default=3, help="Number of layers")
    parser.add_argument("--reg-lambda", '-reg', type=float, default=1., help="Regularization lambda")
    parser.add_argument("--num-epochs", '-epochs', type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch-size", '-bs', type=int, default=128, help="Batch size")
    parser.add_argument("--temperature-optim-lr", '-tlr', type=float, default=5e-3, help="Temperature optimizer learning rate")
    parser.add_argument("--discriminator-optim-lr", '-dlr', type=float, default=5e-3, help="Discriminator optimizer learning rate")
    parser.add_argument("--seed", '-s', type=int, default=0, help="Random seed")
    parser.add_argument("--eval-every", '-e', type=int, default=10, help="Evaluate every n epochs")
    parser.add_argument("--dropout", '-drop', type=float, default=0.05, help="Dropout")
    parser.add_argument("--save-embedding", '-se', action='store_true', help="Save embedding")
    parser.add_argument("--hidden-size", '-hs', type=int, default=None, help="Hidden size")
    parser.add_argument("--log-name", '-log', type=str, default=None, help="Log name")


    args = parser.parse_args()
    args_dict = vars(args)
    output = args_dict.pop('output')

    args_dict['train_adata'] = read_h5ad(args_dict.pop('train_anndata_path'))
    args_dict['test_adata'] = read_h5ad(args_dict.pop('test_anndata_path'))

    args_dict['temperature_transformer'] = joblib.load(args_dict.pop('temperature_transformer'))
    args_dict['homolog_transformer'] = joblib.load(args_dict.pop('homolog_transformer'))

    # Call the main function with the parsed arguments
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        models = main(**args_dict)
    
    # Save the models
    torch.save(models, output)



