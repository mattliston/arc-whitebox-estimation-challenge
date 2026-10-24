"""Fixed CP-free two-memory capacity model; no data access or execution."""

SEED = 9062047


def make_model(torch):
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 2))
    with torch.no_grad():
        model[4].weight.zero_()
        model[4].bias.zero_()
    return model


def step(torch, model, previous, V, rho, Kstar, alpha, gate, pdf, layer,
         *, memory=True):
    """All inputs are causal. No current teacher target/head enters this call."""
    if memory and layer:
        U = V.T @ previous @ V
    else:
        U = torch.zeros_like(Kstar)
    kd, ud = torch.diagonal(Kstar), torch.diagonal(U)
    values = (alpha[:, None], alpha[None, :], gate[:, None], gate[None, :],
              pdf[:, None], pdf[None, :], rho, Kstar, Kstar.T, U, U.T,
              kd[:, None], kd[None, :], ud[:, None], ud[None, :],
              torch.full_like(Kstar, layer / 15.))
    x = torch.stack(torch.broadcast_tensors(*values), dim=-1).clamp(-8, 8)
    decoded = model(x)
    predicted = Kstar + .05 * torch.tanh(decoded[..., 0])
    if memory:
        rho2 = rho * rho
        next_state = torch.tanh(U + rho2 - torch.diag_embed(torch.diagonal(rho2))
                                + decoded[..., 1])
    else:
        next_state = torch.zeros_like(Kstar)
    return predicted, next_state


def step_numpy(np, params, previous, V, rho, Kstar, alpha, gate, pdf, layer,
               *, memory=True):
    """Independent explicit reference, offline only; no Torch dependency."""
    U = V.T @ previous @ V if memory and layer else np.zeros_like(Kstar)
    kd, ud = np.diag(Kstar), np.diag(U)
    values = [alpha[:, None], alpha[None, :], gate[:, None], gate[None, :],
              pdf[:, None], pdf[None, :], rho, Kstar, Kstar.T, U, U.T,
              kd[:, None], kd[None, :], ud[:, None], ud[None, :],
              np.full_like(Kstar, layer / 15.)]
    x = np.stack([np.broadcast_to(a, Kstar.shape) for a in values], axis=-1)
    x = np.clip(x, -8, 8)
    for layer_name in ("0", "2", "4"):
        x = x @ params[layer_name + ".weight"].T + params[layer_name + ".bias"]
        if layer_name != "4":
            x = np.maximum(x, 0)
    predicted = Kstar + .05 * np.tanh(x[..., 0])
    if memory:
        rho2 = rho * rho
        next_state = np.tanh(U + rho2 - np.diag(np.diag(rho2)) + x[..., 1])
    else:
        next_state = np.zeros_like(Kstar)
    return predicted, next_state
