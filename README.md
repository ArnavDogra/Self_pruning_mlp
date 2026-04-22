# The Self-Pruning Neural Network ✂️🧠

**Author:** Arnav Dogra

This repository contains a custom PyTorch implementation of a multi-layer perceptron (MLP) that dynamically prunes its own weights during training. 

By augmenting standard linear layers with learnable gating parameters and applying L1 regularization, the network learns to identify and sever its weakest connections on the fly, optimizing its architecture for deployment in resource-constrained environments.

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone [https://github.com/ArnavDogra/Self_pruning_mlp.git](https://github.com/ArnavDogra/Self_pruning_mlp.git)
cd Self_pruning_mlp 
```
2. Install dependencies:

Bash
pip install -r requirements.txt
3. Run the training script:

Bash
python self_pruning_network.py
(Note: The CIFAR-10 dataset will be automatically downloaded to a ./data directory upon first run).

🔬 The Sparsity Regularization Mechanism
Why an L1 penalty on the sigmoid gates encourages sparsity:

To induce pruning, the network is trained with a composite loss function: Total Loss = CrossEntropyLoss + lambda * SparsityLoss.

The SparsityLoss is defined as the L1 norm (sum of absolute values) of all sigmoid-activated gate scores. Unlike L2 regularization—which has a vanishing gradient near zero and tends to just make weights "small"—the L1 penalty applies a constant sub-gradient. This exerts a persistent, uniform force that pushes gate values toward exactly zero.

If a specific weight's contribution to minimizing the classification loss does not outweigh this constant L1 penalty, the optimizer drives its corresponding gate to 0.0. Because the gates are bound between 0 and 1 via the sigmoid function, this creates a highly polarized, bimodal distribution: critical gates remain near 1.0, while extraneous gates collapse to 0, effectively pruning the connection.

📊 Experimental Results
The network was trained on CIFAR-10 across a sweep of lambda values to observe the trade-off between model sparsity (defined as gates < 0.01) and test accuracy.

Regularization Strength (lambda)	Test Accuracy (%)	Sparsity Level (%)
0.0 (Baseline)	54.88	0.00
1e-05	55.99	1.35
1e-04	55.89	30.38
1e-03	51.84	92.83
Visualizing the Trade-off
As demonstrated below, the network successfully acts as an effective regularizer at moderate lambda values (actually slightly improving accuracy while dropping 30% of weights), and achieves extreme compression at higher values (dropping >92% of weights with minimal accuracy loss).

Gate Distribution Analysis
The log-scaled histogram below (lambda = 1e-4) visually confirms the success of the L1 penalty. The massive spike at zero represents the successfully pruned gates, while the surviving structural connections cluster away from zero.
