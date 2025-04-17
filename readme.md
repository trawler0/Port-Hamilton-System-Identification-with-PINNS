# Nonlinear Port‑Hamiltonian System Identification from Input–State–Output Data

![pipeline](docs/figures/ph-pinns-banner.png)

> **TL;DR** This repository contains the code accompanying the paper **“Non‑linear port‑Hamiltonian system identification from input–state–output data”**.  
> We learn conserved‐energy models directly from trajectories by combining Physics‑Informed Neural Networks (PINNs) with port‑Hamiltonian (pH) structure.

---

## 1. Overview
Port‑Hamiltonian (pH) systems provide an energy–based description of multi‑physical processes and obey

\[\dot x =\bigl(J(x) - R(x)\bigr) \, \nabla_x H(x) + B(x) \, u, \quad y = B^\top(x)\, \nabla_x H(x)\]

where

* **state** \(x \in \mathbb R^{n}\)
* **input** \(u \in \mathbb R^{m}\)
* **output** \(y \in \mathbb R^{m}\)  
* **Hamiltonian** (total energy) **H(x)**  
* **skew‑symmetric interconnection** **J(x) = -J^\top(x)**  
* **positive‑semidefinite dissipation** **R(x) = R^\top(x) \succeq 0**  
* **input matrix** **B(x)**.

Our goal is to identify \(J, R, H, B\) **without explicit supervision** using only the measured triples \((u(t), x(t), y(t))\).  We achieve this by:

* embedding the pH constraints into the loss of a neural model (``Model``),
* exploiting **modular priors** – e.g. *quadratic* or *partially known* Hamiltonians, sparse dissipation, etc.,
* training with the PINNs approach so that the model simultaneously fits \(\dot x\) and \(y\).

The resulting surrogate is **physically consistent**, extrapolates better than black‑box networks, and – in many cases – recovers the true parameters.

---

## 2. Repository structure
```
Port‑Hamilton-System-Identification-with-PINNS/
├─ model/              # model and priors
├─ kan/                # implementation of KAN to compare with MLP
├─ data/               # example datasets (generated on first run)
├─ train/              # training loop in pytorch-lightning
├─ utils/              # collection of utility functions for metrics and forecasting
├─ scripts/            # shell scripts to reproduce all paper figures
└─ README.md           
```

---

## 3. Requirements
* Python
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/)
* [MLflow](https://mlflow.org/)


## 4. Quick start
```bash
# clone repository
$ git clone https://github.com/trawler0/Port-Hamilton-System-Identification-with-PINNS.git
$ cd Port-Hamilton-System-Identification-with-PINNS

# run the all examples
$ bash scripts/run_all.sh            # runs all the examples to reproduce paper results
```
It is also possible to run specific experiments such as recipe.sh
Use results.py to create the figures from the paper. (Need to run the corresponding experiments first. For example, run recipe.sh before calling the recipe() function in results.py.)

## 6. Using your own data
1. **Create a dataset class** in `data.py` that returns `(u, x, y)` tensors.
2. **Modify dim_bias_scale_sigs and simple_experiments functions**
3. **Write a shell script** for your purpose

---

## 7. Citation
If you find this code useful in your research, please cite:
```text
@article{cherifi2025nonlinear,
  title={Nonlinear port-Hamiltonian system identification from input-state-output data},
  author={Cherifi, Karim and Messaoudi, Achraf El and Gernandt, Hannes and Roschkowski, Marco},
  journal={arXiv preprint arXiv:2501.06118},
  year={2025}
}
```

---

## 8. References
*  C. Neary, U. Topcu, Compositional learning of dynamical system mod-
els using port- Hamiltonian neural networks, in: Proceedings of Machine
Learning Research vol 211:1–17, 2023 5th Annual Conference on Learn-
ing for Dynamics and Control, 2023.
*  J. Rettberg, J. Kneifl, J. Herb, P. Buchfink, J. Fehr, B. Haasdonk,
Data-driven identification of latent port-Hamiltonian systems, in: arXiv
preprint arXiv:2408.03212, 2024.
* A. Desai, L. Li, I. Chakrabarty, C. Bajaj, S. Gupta, Port-Hamiltonian
neural networks for constrained mechanical systems, in: arXiv preprint
arXiv:2106.13188, 2021.
---

## 9. Contact
For questions, feel free to open an issue or contact **Trawler O.**:<br>
<mailto:trawler@example.com>

---

