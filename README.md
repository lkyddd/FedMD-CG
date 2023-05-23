# FedMD-CG

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.9.0

- scipy==1.7.2

- numpy==1.21.2

- sklearn==1.0.1

- matplotlib==3.5.3

- pandas==1.3.4

- mpi4py==3.1.1

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- LightFed # experiments for baselines, FedMD-CG and datasets
    |-- experiments/ 
        |-- datasets/ 
            |-- data_distributer.py/  # the load datasets,including MNIST, EMNIST, FMNSIT and CIFAR-10
        |-- horizontal/ ## FedMD-CG and baselines
            |-- fedavg/
            |-- FedCG/
            |-- fedgen/
            |-- FedMD-CG/
            |-- Fedper/
            |-- LG-Fedavg/
            |-- localtrain/
        |-- models
            |-- model.py/  ##load backnone architectures
    |-- lightfed/  
        |-- core # important configure
        |-- tools
```

## Run pipeline for Run pipeline for FedMD-CG
1. Entering the FedMD-CG
```python
cd LightFed
cd experiments
cd horizontal
cd FedMD-CG
```

2. You can run any models implemented in `main_FedMDCG.py`. For examples, you can run our model on `FMNIST` dataset by the script:
```python
python main_FedMDCG.py --data_partition_mode non_iid_dirichlet_balanced --non_iid_alpha 10 --client_num 10 --seed 0 --model_type Lenet --data_set FMNIST
```
And you can run other baselines, such as 
```python
cd LightFed
cd experiments
cd horizontal
cd fedavg
python main_fedavg.py --data_partition_mode non_iid_dirichlet_balanced --non_iid_alpha 10 --client_num 10 --seed 0 --model_type Lenet --data_set FMNIST
```

