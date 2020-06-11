import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
#logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    loss_1_nograd=loss_1.cpu().detach().numpy()
    ind_1_sorted = np.argsort(loss_1_nograd) # sort the loss 
    loss_1_sorted = loss_1[ind_1_sorted]



    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    loss_2_nograd=loss_2.cpu().detach().numpy()
    ind_2_sorted = np.argsort(loss_2_nograd)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_2_sorted))
    
    ind_1=ind_1_sorted[:num_remember]
    ind_2=ind_2_sorted[:num_remember]
    noise_ind_1=ind[ind_1]
    noise_ind_2=ind[ind_2]
    
    pure_ratio_1 = np.sum(noise_or_not[noise_ind_1])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[noise_ind_2])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    y_1_update=y_1[ind_2_update]
    y_2_update=y_2[ind_1_update]
    t_1_update=t[ind_2_update]
    t_2_update=t[ind_1_update]
    loss_1_update = F.cross_entropy(y_1_update, t_1_update)
    loss_2_update = F.cross_entropy(y_2_update, t_2_update)
# loss average on first model, loss average on second model, pure ratio on first--> to what percent is the remembered sample clean , pure ratio on second
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


