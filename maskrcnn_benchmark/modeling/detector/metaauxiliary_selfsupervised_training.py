import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchmeta.modules import MetaModule

def gradient_update_parameters(model, loss, params=None, step_size=1., first_order=False):

    if not isinstance(model, MetaModule):
       raise ValueError('The model must be an instance of torchmeta.modules.MetaModule, got {}'.format(type(model)))

    if params is None:
       params = OrderedDict(model.meta_named_parameters())
       #print(params.keys())
    #pdb.set_trace()
    grads = torch.autograd.grad(loss, [value for name, value in params.items() if value.requires_grad == True], create_graph=not first_order, allow_unused=True)
    #print(grads)

    updated_params = OrderedDict()

    i = 0
    for name, param in params.items():
       if param.requires_grad == False:
          updated_params[name] = param
          continue
       try:
           updated_params[name] = param - step_size[0] * grads[i]
       except TypeError as e:
           #print("Except {}: Nome parametro :{}".format(e, name))
           updated_params[name] = param
       i = i + 1
    return updated_params
