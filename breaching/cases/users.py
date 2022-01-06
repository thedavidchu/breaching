"""Implement user code."""

import torch
import copy
from itertools import chain

from .data import construct_dataloader


def construct_user(model, loss_fn, cfg_case, setup):
    """Interface function."""
    if cfg_case.user.user_type == "local_gradient":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg_case.user.user_idx)
        # The user will deepcopy this model template to have their own
        user = UserSingleStep(model, loss_fn, dataloader, setup, idx=cfg_case.user.user_idx, cfg_user=cfg_case.user)
    elif cfg_case.user.user_type == "local_update":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg_case.user.user_idx)
        user = UserMultiStep(model, loss_fn, dataloader, setup, idx=cfg_case.user.user_idx, cfg_user=cfg_case.user)
    elif cfg_case.user.user_type == "multiuser_aggregate":
        dataloaders = []
        for idx in range(*cfg_case.user.user_range):
            dataloaders += [construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=idx)]
        user = MultiUserAggregate(model, loss_fn, dataloaders, setup, cfg_case.user)
    return user


class UserSingleStep(torch.nn.Module):
    """A user who computes a single local update step."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize from cfg_user dict which contains atleast all keys in the matching .yaml :>"""
        super().__init__()
        self.num_data_points = cfg_user.num_data_points

        self.provide_labels = cfg_user.provide_labels
        self.provide_num_data_points = cfg_user.provide_num_data_points
        self.provide_buffers = cfg_user.provide_buffers

        self.user_idx = idx
        self.setup = setup

        self.model = copy.deepcopy(model)
        self.model.to(**setup)

        self.defense_repr = []
        self._initialize_local_privacy_measures(cfg_user.local_diff_privacy)

        self.dataloader = dataloader
        self.loss = copy.deepcopy(loss)  # Just in case the loss contains state

    def __repr__(self):
        n = "\n"
        return f"""User (of type {self.__class__.__name__}) with settings:
    Number of data points: {self.num_data_points}

    Threat model:
    User provides labels: {self.provide_labels}
    User provides buffers: {self.provide_buffers}
    User provides number of data points: {self.provide_num_data_points}

    Data:
    Dataset: {self.dataloader.name}
    user: {self.user_idx}
    {n.join(self.defense_repr)}
        """

    def _initialize_local_privacy_measures(self, local_diff_privacy):
        """Initialize generators for noise in either gradient or input."""
        if local_diff_privacy["gradient_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["gradient_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} gradient noise with strength {scale.item()}.'
            )
        else:
            self.generator = None
        if local_diff_privacy["input_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["input_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator_input = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator_input = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} input noise with strength {scale.item()}.'
            )
        else:
            self.generator_input = None
        self.clip_value = local_diff_privacy.get("per_example_clipping", 0.0)
        if self.clip_value > 0:
            self.defense_repr.append(f"Defense: Gradient clipping to maximum of {self.clip_value}.")

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload.

        Batchnorm behavior:
        If public buffers are sent by the server, then the user will be set into evaluation mode
        Otherwise the user is in training mode and sends back buffer based on .provide_buffers.

        Shared labels are canonically sorted for simplicity."""

        data = self._load_data()
        B = data["labels"].shape[0]
        # Compute local updates
        shared_grads = []
        shared_buffers = []

        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                for module in self.model.modules():
                    if hasattr(module, "momentum"):
                        module.momentum = None  # Force recovery without division
                self.model.train()

        def _compute_batch_gradient(data):
            data["inputs"] = (
                data["inputs"] + self.generator_input.sample(data["inputs"].shape)
                if self.generator_input is not None
                else data["inputs"]
            )
            outputs = self.model(**data)
            loss = self.loss(outputs, data["labels"])
            return torch.autograd.grad(loss, self.model.parameters())

        if self.clip_value > 0:  # Compute per-example gradients and clip them in this case
            shared_grads = [torch.zeros_like(p) for p in self.model.parameters()]
            for data_idx in range(B):
                data_point = {key: val[data_idx : data_idx + 1] for key, val in data.items()}
                per_example_grads = _compute_batch_gradient(data_point)
                self._clip_list_of_grad_(per_example_grads)
                torch._foreach_add_(shared_grads, per_example_grads)
            torch._foreach_div_(shared_grads, B)
        else:
            # Compute the forward pass
            shared_grads = _compute_batch_gradient(data)
        self._apply_differential_noise(shared_grads)

        if buffers is not None:
            shared_buffers = None
        else:
            shared_buffers = [b.clone().detach() for b in self.model.buffers()]

        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=data["labels"].sort()[0] if self.provide_labels else None,
            local_hyperparams=None,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=data["inputs"], labels=data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data

    def _clip_list_of_grad_(self, grads):
        """Apply differential privacy component per-example clipping."""
        grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        if grad_norm > self.clip_value:
            [g.mul_(self.clip_value / (grad_norm + 1e-6)) for g in grads]

    def _apply_differential_noise(self, grads):
        """Apply differential privacy component gradient noise."""
        if self.generator is not None:
            for grad in grads:
                grad += self.generator.sample(grad.shape)

    def _load_data(self):
        """Generate data from dataloader, truncated by self.num_data_points"""
        # Select data
        data_blocks = []
        num_samples = 0

        for idx, data_block in enumerate(self.dataloader):
            data_blocks += [data_block]
            num_samples += data_block["labels"].shape[0]
            if num_samples > self.num_data_points:
                break

        if num_samples < self.num_data_points:
            raise ValueError(
                f"This user does not have the requested {self.num_data_points} samples,"
                f"they only own {num_samples} samples."
            )

        data = dict()
        for key in data_blocks[0]:
            data[key] = torch.cat([d[key] for d in data_blocks], dim=0)[: self.num_data_points].to(
                device=self.setup["device"]
            )
        return data

    def print(self, user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer
        decoded_tokens = tokenizer.batch_decode(user_data["data"], clean_up_tokenization_spaces=True)
        for line in decoded_tokens:
            print(line)

    def plot(self, user_data, scale=False, print_labels=False):
        """Plot user data to output. Probably best called from a jupyter notebook."""
        import matplotlib.pyplot as plt  # lazily import this here

        dm = torch.as_tensor(self.dataloader.dataset.mean, **self.setup)[None, :, None, None]
        ds = torch.as_tensor(self.dataloader.dataset.std, **self.setup)[None, :, None, None]
        classes = self.dataloader.dataset.classes

        data = user_data["data"].clone().detach()
        labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
        if labels is None:
            print_labels = False

        if scale:
            min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
            # print(f'min_val: {min_val} | max_val: {max_val}')
            data = (data - min_val) / (max_val - min_val)
        else:
            data.mul_(ds).add_(dm).clamp_(0, 1)
        data = data.to(dtype=torch.float32)

        if data.shape[0] == 1:
            plt.axis("off")
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            if print_labels:
                plt.title(f"Data with label {classes[labels]}")
        else:
            grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
            s = 24 if data.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(data, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if labels is not None and print_labels:
                    label_classes.append(classes[labels[i]])
                axis.axis("off")
            if print_labels:
                print(label_classes)


class UserMultiStep(UserSingleStep):
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, idx, cfg_user)

        self.num_local_updates = cfg_user.num_local_updates
        self.num_data_per_local_update_step = cfg_user.num_data_per_local_update_step
        self.local_learning_rate = cfg_user.local_learning_rate
        self.provide_local_hyperparams = cfg_user.provide_local_hyperparams

    def __repr__(self):
        n = "\n"
        return (
            super().__repr__()
            + n
            + f"""    Local FL Setup:
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""

        user_data = self._load_data()

        # Compute local updates
        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
        seen_data_idx = 0
        label_list = []
        for step in range(self.num_local_updates):
            data = {
                k: v[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step] for k, v in user_data.items()
            }
            seen_data_idx += self.num_data_per_local_update_step
            seen_data_idx = seen_data_idx % self.num_data_points
            label_list.append(data["labels"].sort()[0])

            optimizer.zero_grad()
            # Compute the forward pass
            data["inputs"] = (
                data["inputs"] + self.generator_input.sample(data["inputs"].shape)
                if self.generator_input is not None
                else data["inputs"]
            )
            outputs = self.model(**data)
            loss = self.loss(outputs, data["labels"])
            loss.backward()

            grads_ref = [p.grad for p in self.model.parameters()]
            if self.clip_value > 0:
                self._clip_list_of_grad_(grads_ref)
            self._apply_differential_noise(grads_ref)
            optimizer.step()

        # Share differential to server version:
        # This is equivalent to sending the new stuff and letting the server do it, but in line
        # with the gradients sent in UserSingleStep
        shared_grads = [
            (p_local - p_server.to(**self.setup)).clone().detach()
            for (p_local, p_server) in zip(self.model.parameters(), parameters)
        ]

        shared_buffers = [b.clone().detach() for b in self.model.buffers()]
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_data["labels"] if self.provide_labels else None,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
            data_key="inputs",
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=user_data["inputs"], labels=user_data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data


class MultiUserAggregate(UserMultiStep):
    """A silo of users who compute local updates as in a fedSGD or fedAVG scenario and aggregate their results.

    For an unaggregated single silo refer to SingleUser classes as above.
    This aggregration is assumed to be safe (e.g. via secure aggregation) and the attacker and server only gain
    access to the aggregated local updates.

    self.dataloader of this class is actually quite unwieldy, due to its possible size.
    For the same reason the true_user_data that is returned contains only references to all dataloaders.
    """

    def __init__(self, model, loss, dataloader, setup, cfg_user):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, None, cfg_user)

        self.num_users = len(dataloader)

        self.users = []
        self.user_setup = dict(dtype=setup["dtype"], device=torch.device("cpu"))  # Simulate on CPU
        for idx in range(self.num_users):
            if self.num_local_updates > 1:
                self.users.append(UserMultiStep(model, loss, dataloader[idx], self.user_setup, idx, cfg_user))
            else:
                self.users.append(UserSingleStep(model, loss, dataloader[idx], self.user_setup, idx, cfg_user))

        self.dataloader = chain(*dataloader)

    def __repr__(self):
        n = "\n"
        return self.users[0].__repr__() + n + f"""    Number of aggregated users: {self.num_users}"""

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""
        # Compute local updates

        server_parameters = server_payload["parameters"]
        server_buffers = server_payload["buffers"]

        aggregate_updates = [torch.zeros_like(p) for p in self.model.parameters()]
        aggregate_buffers = [torch.zeros_like(b, dtype=torch.float) for b in self.model.buffers()]
        aggregate_labels = []
        aggregate_label_lists = []  # Only ever used in rare sanity checks. List of labels per local update step
        for user in self.users:
            user.to(**self.setup)
            user_data, _ = user.compute_local_updates(server_payload)
            user.to(**self.user_setup)

            torch._foreach_sub_(user_data["gradients"], aggregate_updates)
            torch._foreach_add_(aggregate_updates, user_data["gradients"], alpha=-1 / self.num_users)

            if user_data["buffers"] is not None:
                torch._foreach_sub_(user_data["buffers"], aggregate_buffers)
                torch._foreach_add_(aggregate_buffers, buffer_to_server, alpha=1 / self.num_users)
            if user_data["metadata"]["labels"] is not None:
                aggregate_labels.append(user_data["labels"].cpu())
            if params := user_data["metadata"]["local_hyperparams"] is not None:
                if params["labels"] is not None:
                    aggregate_label_lists += [l.cpu() for l in user_data["metadata"]["local_hyperparams"]["labels"]]

        shared_data = dict(
            gradients=aggregate_updates,
            buffers=aggregate_buffers if self.provide_buffers else None,
            metadata=dict(
                num_data_points=self.num_data_points if self.provide_num_data_points else None,
                labels=torch.cat(aggregate_labels.sort()[0]) if self.provide_labels else None,
                num_users=self.num_users,
                local_hyperparams=dict(
                    lr=self.local_learning_rate,
                    steps=self.num_local_updates,
                    data_per_step=self.num_data_per_local_update_step,
                    labels=aggregate_label_lists,
                )
                if self.provide_local_hyperparams
                else None,
            ),
        )

        def generate_user_data():
            for user in range(self.users):
                yield user._load_data()

        true_user_data = dict(data=generate_user_data(), labels=None, buffers=aggregate_buffers)

        return shared_data, true_user_data
