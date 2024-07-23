import numpy as np
import torch

def format(label, array):

    prefix = label + ' = (float []) '
    txt = np.array2string(array.flatten(), prefix=prefix, separator='f,', sign='+', floatmode='fixed')
    txt = txt.replace('[', '{').replace(']', 'f}')
    txt = prefix + txt

    return txt

def export(layer):

    txt = ''

    if isinstance(layer, torch.nn.modules.rnn.GRU):
        
        txt += '.num_dims_in = %u,\n' % int(layer.weight_ih_l0.shape[1])
        txt += '.num_dims_out = %u,\n' % int(layer.weight_ih_l0.shape[0]/3)
        txt += format('.W_ih', layer.weight_ih_l0.detach().numpy()) + ",\n"
        txt += format('.W_hh', layer.weight_hh_l0.detach().numpy()) + ",\n"
        txt += format('.b_ih', layer.bias_ih_l0.detach().numpy()) + ",\n"
        txt += format('.b_hh', layer.bias_hh_l0.detach().numpy()) + "\n"

    if isinstance(layer, torch.nn.modules.rnn.LSTM):

        txt += '.num_dims_in = %u,\n' % int(layer.weight_ih_l0.shape[1])
        txt += '.num_dims_out = %u,\n' % int(layer.weight_ih_l0.shape[0]/4)
        txt += format('.W_ih', layer.weight_ih_l0.detach().numpy()) + ",\n"
        txt += format('.W_hh', layer.weight_hh_l0.detach().numpy()) + ",\n"
        txt += format('.b_ih', layer.bias_ih_l0.detach().numpy()) + ",\n"
        txt += format('.b_hh', layer.bias_hh_l0.detach().numpy()) + "\n"

    return txt

def main():

    torch.manual_seed(1)

    x = torch.randn((1,2,4))
    h = torch.randn((1,2,3))
    c = torch.randn((1,2,3))

    layer = torch.nn.LSTM(4, 3)
    
    h_target, c_target = layer(x, (h, c))[1]

    print(export(layer))
    print(format('in_array[]', x.detach().numpy()))
    print(format('hidden_in_array[]', h.detach().numpy()))
    print(format('cell_in_array[]', c.detach().numpy()))
    print(format('hidden_target_array[]', h_target.detach().numpy()))
    print(format('cell_target_array[]', c_target.detach().numpy()))

    return 0

if __name__ == '__main__':

    main()