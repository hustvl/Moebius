"""
copied from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/ema.py#L5
"""

from contextlib import contextmanager
import torch
from torch import nn


@contextmanager
def ema_scope(use_ema, model_ema, model, context=None):
    if use_ema:
        model_ema.store(model.parameters())
        model_ema.copy_to(model)
        if context is not None:
            print(f"{context}: Switched to EMA weights")
    try:
        yield None
    finally:
        if use_ema:
            model_ema.restore(model.parameters())
            if context is not None:
                print(f"{context}: Restored training weights")

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


def save_litema(ema_model, filepath):
    torch.save({
        "ema_buffers": ema_model.state_dict(),  
        "ema_name_map": ema_model.m_name2s_name, 
    }, filepath)



def load_litema(ema_model, filepath, strict=True, map_location='cpu'):
    ckpt = torch.load(filepath, map_location=map_location)
    msg = ema_model.load_state_dict(ckpt["ema_buffers"], strict=strict)
    ema_model.m_name2s_name = ckpt["ema_name_map"]
    return msg


def load_ema_into_model(model, filepath, decay=0.9999):
    ema = LitEma(model, decay=decay)
    load_litema(ema, filepath)
    ema.copy_to(model)
    return model
