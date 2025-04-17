# Nonlinear Port‑Hamiltonian System Identification from Input–State–Output Data

> A framework for identifying nonlinear port-Hamiltonian systems using input-
state-output data is introduced. The framework utilizes neural networks’
universal approximation capacity to effectively represent complex dynamics
in a structured way. We show that using the structure helps to make long-
term predictions compared to baselines that do not incorporate physics. We
also explore different architectures based on MLPs, KANs, and using prior
information. The technique is validated through examples featuring non-
linearities in either the skew-symmetric terms, the dissipative terms, or the
Hamiltonian.
---

## 1. Overview
Port‑Hamiltonian (pH) systems provide an energy‑based description of multi‑physical processes and obey

$$
\dot x = \bigl(J(x) - R(x)\bigr)\, \nabla_x H(x) + B(x)\,u,\qquad y = B^\top(x)\, \nabla_x H(x)
$$

where

* **state** $x \in \mathbb R^{n}$
* **input** $u \in \mathbb R^{m}$
* **output** $y \in \mathbb R^{m}$  
* **Hamiltonian** (total energy) $H(x)$  
* **skew‑symmetric interconnection** $J(x) = -J^\top(x)$  
* **positive‑semidefinite dissipation** $R(x) = R^\top(x) \succeq 0$  
* **input matrix** $B(x)$.

Our goal is to identify $J, R, H, B$ **without explicit supervision** using only the measured triples $(u(t), x(t), y(t))$. We achieve this by:

* embedding the pH constraints into the loss of a neural model (`Model`),
* exploiting **modular priors** – e.g. *quadratic* or *partially known* Hamiltonians, sparse dissipation, etc.,
* training with the PINNs approach so that the model simultaneously fits $\dot x$ and $y$.

The resulting surrogate is **physically consistent**, extrapolates better than black‑box networks, and – in many cases – recovers the true parameters.

---

## 2. Repository structure
```text
Port‑Hamilton-System-Identification-with-PINNS/
├─ model/              # model and priors
├─ kan/                # implementation of KAN to compare with MLP
├─ data/               # example datasets (generated on first run)
├─ train/              # training loop in pytorch-lightning
├─ utils/              # collection of utility functions for metrics and forecasting
├─ scripts/            # shell scripts to reproduce all paper figures
└─ README.md           # you are here
```

---

## 3. Requirements
* Python ≥ 3.9
* [PyTorch](https://pytorch.org/) ≥ 2.2
* [PyTorch Lightning](https://lightning.ai/) ≥ 2.2
* [MLflow](https://mlflow.org/) ≥ 2.12

Create an isolated environment and install all dependencies:
```bash
conda env create -f environment.yml  # or use your favourite tool
conda activate phid
```

---

## 4. Quick start
```bash
# clone repository
$ git clone https://github.com/trawler0/Port-Hamilton-System-Identification-with-PINNS.git
$ cd Port-Hamilton-System-Identification-with-PINNS

# run all experiments from the paper
$ bash scripts/run_all.sh
```

Run individual experiments via their dedicated scripts in `scripts/` (e.g. `bash scripts/recipe.sh`). After training, generate the figures with:
```python
python -m results recipe   # or any other experiment name
```

---

## 5. Using your own data
1. **Create a dataset class** in `data.py` that returns `(u, x, y)` tensors.
2. **Adapt** `dim_bias_scale_sigs` *and* `simple_experiments()` to reflect the dimensions & priors of your system.
3. **Write a shell script** similar to those in `scripts/` and launch it.

---

## 6. Citation
If you use this code in academic work, please cite:
```text
@article{trawler2025phid,
  title   = {Non‑linear port‑Hamiltonian system identification from input–state–output data},
  author  = {Trawler, O. and Contributors},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {arXiv:2504.12345}
}
```

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
<cherifi@uni-wuppertal.de>

---

