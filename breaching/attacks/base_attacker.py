"""Implementation for base attacker class.

Inherit from this class for a consistent interface with attack cases."""

import torch
from collections import defaultdict
import copy

class _BaseAttacker():
    """This is a template class for an attack."""

    def __init__(model, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        self.cfg = cfg_attack
        self.setup = setup
        self.model_template = copy.deepcopy(model)

    def reconstruct(self, server_payload, shared_data):

        stats = defaultdict(list)

        # Implement the attack here
        # The attack should consume the shared_data and server payloads and reconstruct
        raise NotImplementedError()

        return reconstructed_data, stats

    def _construct_models_from_payload(self, server_payload):
        """Construct the model (or multiple) that is sent by the server."""

        # Load states into multiple models if necessary
        models = []
        for payload in server_payload:
            parameters = payload['parameters']
            buffers = payload['buffers']
            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup)

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
            models.append(new_model)
        return models

    def _initialize_data(self, data_shape):
        init_type = self.cfg.init
        if init_type == 'randn':
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == 'rand':
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == 'zeros':
            candidate = torch.zeros(data_shape, **self.setup)

        candidate.requires_grad = True
        return candidate

    def _init_optimizer(self, candidate):
        optim_name = cfg.optim.optimizer
        if optim_name == 'adam':
            optimizer = torch.optim.Adam(candidate, lr=cfg.optim.step_size)
        elif optim_name == 'momGD':
            optimizer = torch.optim.SGD(candidate, lr=cfg.optim.step_size, momentum=0.9, nesterov=True)
        elif optim_name == 'GD':
            optimizer = torch.optim.SGD(candidate, lr=cfg.optim.step_size, momentum=0.0)
        elif optim_name == 'L-BFGS':
            optimizer = torch.optim.LBFGS(candidate, lr=cfg.optim.step_size)
        else:
            raise ValueError(f'Invalid optimizer {optim_name} given.')

        if cfg.optim.step_size_decay:
            max_iterations = cfg.optim.max_iterations
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                                                         max_iterations // 1.142], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)

        return optimizer, scheduler


    def _recover_label_information(self, user_data):
        raise NotImplementedError()
