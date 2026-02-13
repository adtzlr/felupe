r"""
Train a deep neural network
---------------------------

.. topic:: Train a PyTorch model by simulation results.

   * compression of a hyperelastic axisymmetric cylinder

   * evaluate displacements located at mesh-points

   * train a neural network on the displacement data

   * obtain displacements from the PyTorch model and plot the log. strains

First, an axisymmetric model is created. The displacements are saved after each
completed substep. Only very few substeps are used to run the simulation.

.. admonition:: This example requires external packages.
   :class: hint

   .. code-block::

      pip install torch
"""

# sphinx_gallery_thumbnail_number = -1
import numpy as np

import felupe as fem

mesh = fem.Rectangle(a=(0, 1), b=(1, 3), n=(11, 31))
region = fem.RegionQuad(mesh)
field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
boundaries = fem.dof.uniaxial(field, clamped=True, sym=False, return_loadcase=False)
solid = fem.SolidBody(fem.NeoHookeCompressible(mu=1, lmbda=10), field)
move = fem.math.linsteps([0, -0.2], num=3)
step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
displacements = np.zeros((len(move), *field[0].values.shape))


def save(i, j, res):
    displacements[j] = res.x[0].values


job = fem.Job(steps=[step], callback=save).evaluate(tol=1e-1)

# %%
# A PyTorch model is trained on the simulation results. For simplicity, testing is
# skipped and the data is not splitted in batches.
import torch
import torch.nn as nn
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.reshape(len(x), mesh.npoints, mesh.dim)


xv, yv = move.reshape(-1, 1), displacements
x = torch.tensor(xv, dtype=torch.float32)
y = torch.tensor(yv, dtype=torch.float32)

model = NeuralNetwork(in_features=xv[0].size, out_features=yv[0].size)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train(x, y, model, loss_fn, optimizer, verbose=False):
    model.train()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if verbose:
        print(f"Train Loss (Final): {loss.item():.2e}")


def run_epochs(n=100):
    for t in tqdm(range(n - 1)):
        train(x, y, model, loss_fn, optimizer)
    train(x, y, model, loss_fn, optimizer, verbose=True)
    model.eval()


# %%
# Max. principal values of the logarithmic strain tensors are evaluated and plotted
# based on the PyTorch model displacement results. After 50 epoch runs, the result is
# quite bad with a lot of unwanted artefacts (noise).
run_epochs(n=50)
field_2 = field.copy()
field_2[0].values[:] = model(torch.Tensor([[-0.17]])).detach().numpy()[0]
field_2.plot("Principal Values of Logarithmic Strain").show()

# %%
# After 500 more epoch runs, the result is much more realistic.
run_epochs(n=500)
field_2[0].values[:] = model(torch.Tensor([[-0.17]])).detach().numpy()[0]
field_2.plot("Principal Values of Logarithmic Strain").show()
