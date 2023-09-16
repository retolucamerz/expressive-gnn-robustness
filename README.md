## Adversarial Robustness of Expressive Graph Neural Networks

This repository contains the code for my bachelor's thesis.


### Abstract

In recent years, graph neural networks (GNNs) have gained popularity as a tool for machine learning on graphs. Despite their success in many fields, many GNN architectures lack the expressiveness to differentiate between non-isomorphic graphs of certain classes, motivating more expressive architectures. Other works have shown basic GNN architectures and defenses vulnerable to adversarial attacks, which generate adversarial examples that are mispredicted by the model. We investigate how the expressivity of a GNN architecture influences its robustness to adversarial attacks by designing graph-level test-time attacks targeting the discrete structure of graphs with node and edge features. Our experiments indicate that PPGN, the most expressive architecture tested, is more robust on a range of attacks than the other tested architectures, and can even outperform adversarially trained models of other expressive architectures under attacks.


### Environment and Tests

To setup the environment, run the following commands:
```
conda create -n egr python=3.10
conda activate egr
conda install numpy=1.24.3
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg pytorch-scatter -c pyg
pip install ogb==1.3.6 rdkit==2023.3.1 pandas==2.0.2 seaborn==0.12.2 matplotlib==3.7.1 wandb==0.15.4 tensorboard==2.13.0 tqdm==4.65.0
```

Run the tests with `python -m unittest discover tests`. To run a specific test, use `python -m unittest tests/test_....py`.
