import os 
import sys
import datetime
import time 
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')

def argmin(f, x0, lr = 0.001, n_epoch = 1000000):
    # input:
    #   f: func to be optimized, with single variable x
    #   x0: initial value of x
    # Output:
    #   x* = argmin f(x).
    # Param:
    #   delta: the stopping threshold
    #   lr: lreaning rate
    delta = 1e-5
    
    
    x = x0.clone().detach().requires_grad_(True)
    x_pre = x.clone().detach()
    optimizer = torch.optim.SGD([x],lr=lr)
    for i in range(n_epoch):
        optimizer.zero_grad()
        f_x = f(x)
        f_x.backward()
        optimizer.step()
        if i%1000 == 0 and i>1:
            if (x_pre - x).abs().sum() < delta:
                #print("SGD early break, i=%d"%i)
                break
            x_pre = x.clone().detach()
    return x.clone().detach()


def draw(log, title=""):
    # inputs:
    # data=map(string -> list of float), eg. {"obj":[0.1,...], "constr":[0.1,..], .. }
    # titme = str
   
    fig, ax = plt.subplots()
    for name, data in log.items():
        ax.plot(np.arange(len(data)), data, label=name)
    plt.legend()
    plt.xlabel("Iter")
    plt.yscale("log")
    plt.title(title)
    #plt.show()
    if title == "":
        title = str(datetime.datetime.now())
    plt.savefig("%s.png"%title, dpi =200)
    plt.close()