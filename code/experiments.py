import os, argparse
from pathlib import Path
import torch
import pickle, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

os.chdir(Path(__file__).parent.resolve())

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters

from models.baseline.baseline_vae import BaselineVAE

from utils.function_runners import (
    main, 
    handle
)

@handle("baseline_rvae")
def baseline_test_rvae_to_GPT():
    batch_loader = BatchLoader()
    parameters = Parameters(batch_loader.vocab_size)

    vae = BaselineVAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
              parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers)
    
    optimizer = Adam(vae.parameters(), 0.01)

    # num of iterations
    for iteration in range(20):

        '''Train step'''
        # batch size, don't use cuda
        input, decoder_input, target = batch_loader.next_batch(20, 'train', False)
        target = target.view(-1)

        logits, aux_logits, kld = vae(0.12, input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = cross_entropy + 0.4 * aux_cross_entropy + kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Validation'''
        input, decoder_input, target = batch_loader.next_batch(30, 'valid', False)
        target = target.view(-1)

        logits, aux_logits, valid_kld = vae(0.12, input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        valid_cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = valid_cross_entropy + 0.4 * valid_aux_cross_entropy + kld

        if iteration % 50 == 0:
            print('\n')
            print('|--------------------------------------|')
            print(iteration)
            print('|--------ce------aux-ce-----kld--------|')
            print('|----------------train-----------------|')
            print(cross_entropy.data.cpu().numpy()[0]/(210 * 30),
                  aux_cross_entropy.data.cpu().numpy()[0]/(210 * 30),
                  kld.data.cpu().numpy()[0])
            print('|----------------valid-----------------|')
            print(valid_cross_entropy.data.cpu().numpy()[0]/(210 * 30),
                  valid_aux_cross_entropy.data.cpu().numpy()[0]/(210 * 30),
                  valid_kld.data.cpu().numpy()[0])
            print('|--------------------------------------|')
            input, _, _ = batch_loader.next_batch(2, 'valid', False)
            mu, logvar = vae.inference(input[0].unsqueeze(1))
            std = torch.exp(0.5 * logvar)

            z = Variable(torch.randn([1, parameters.latent_size]))
            if False:
                z = z.cuda()
            z = z * std + mu
            print(''.join([batch_loader.idx_to_char[idx] for idx in input.data.cpu().numpy()[0]]))
            print(vae.sample(batch_loader, False, z))
            print('|--------------------------------------|')



if __name__ == "__main__":
    main()