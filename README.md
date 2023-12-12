Installation
------------

```
conda create --name adversarial -c conda-forge python=3.11 mamba -y
conda activate adversarial
mamba install -c pytorch -c conda-forge pytorch anndata joblib tqdm scikit-learn -y
pip install tensorboard
```

Usage
-----

```
python loss_experiment.py <embeddings_file> <run_name> --split-type {group,random}
```

Will save:

* training information to `runs/<run_name>` in tensorboard log files
* an example model to `models/<run_name>.model.pt`
* training and testing loss information to `results/<run_name>.results.tsv`

Split-type controls if the test set is partitioned by domain, or randomly.

This command works with the embeddings I provided.