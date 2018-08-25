import torch.utils as utils

def load_pretrained_model(model, pretrained_state):
    own_state = model.state_dict()
    for key, param in pretrained_state.items():
        if key not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)
