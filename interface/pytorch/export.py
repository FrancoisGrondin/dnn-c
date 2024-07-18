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

    return txt

def main():

    layer = torch.nn.GRU(1, 2, batch_first=True)
    with torch.no_grad():
        
        layer.weight_ih_l0[:,:] = torch.from_numpy(np.zeros((6,1), dtype=np.float32))
        layer.weight_hh_l0[:,:] = torch.from_numpy(np.zeros((6,2), dtype=np.float32))
        layer.bias_ih_l0[:] = torch.from_numpy(np.zeros((6,), dtype=np.float32))
        layer.bias_hh_l0[:] = torch.from_numpy(np.zeros((6,), dtype=np.float32))

        layer.weight_ih_l0[0,0] = +5.0
        layer.weight_ih_l0[1,0] = -3.0
        layer.weight_ih_l0[2,0] = +2.0
        layer.weight_ih_l0[3,0] = +1.0
        layer.weight_ih_l0[4,0] = +1.0
        layer.weight_ih_l0[5,0] = +1.0

        layer.weight_hh_l0[0,0] = +1.0
        layer.weight_hh_l0[0,1] = -2.0
        layer.weight_hh_l0[1,0] = +3.0
        layer.weight_hh_l0[1,1] = -4.0
        layer.weight_hh_l0[2,0] = +5.0
        layer.weight_hh_l0[2,1] = -6.0
        layer.weight_hh_l0[3,0] = +7.0
        layer.weight_hh_l0[3,1] = -8.0
        layer.weight_hh_l0[4,0] = +9.0
        layer.weight_hh_l0[4,1] = +0.0
        layer.weight_hh_l0[5,0] = +1.0
        layer.weight_hh_l0[5,1] = -2.0

        layer.bias_ih_l0[0] = +1.0
        layer.bias_ih_l0[1] = +2.0
        layer.bias_ih_l0[2] = +3.0
        layer.bias_ih_l0[3] = +4.0
        layer.bias_ih_l0[4] = +5.0
        layer.bias_ih_l0[5] = +6.0

        layer.bias_hh_l0[0] = +3.0
        layer.bias_hh_l0[1] = +6.0
        layer.bias_hh_l0[2] = +9.0
        layer.bias_hh_l0[3] = -2.0
        layer.bias_hh_l0[4] = -5.0
        layer.bias_hh_l0[5] = -8.0


    x = torch.zeros((1,1,1), dtype=torch.float32)
    h = torch.zeros((1,1,2), dtype=torch.float32)
    
    x[0, 0, 0] = -4.0
    h[0, 0, 0] = 1.0
    h[0, 0, 1] = 4.0

    print(x)
    print(h)

    y = layer(x, h)

    print(y)

    return 0

if __name__ == '__main__':

    main()