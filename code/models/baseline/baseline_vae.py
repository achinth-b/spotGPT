import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn.init import xavier_normal

from utils.batch_loader import BatchLoader
from .baseline_encoder import BaselineEncoder
from .baseline_decoder import BaselineDecoder


class BaselineVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_size, decoder_size, decoder_num_layers):
        super(BaselineVAE, self).__init__()

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.embed.weight = xavier_normal(self.embed.weight)

        self.encoder = BaselineEncoder(self.embed_size, self.latent_size)

        self.context_to_mu = nn.Linear(self.latent_size, self.latent_size)
        self.context_to_logvar = nn.Linear(self.latent_size, self.latent_size)

        self.decoder = BaselineDecoder(self.vocab_size, self.latent_size, decoder_size, decoder_num_layers, self.embed_size)

    def forward(self, drop_prob,
                encoder_input=None,
                decoder_input=None,
                z=None):
        """
        :param drop_prob: Probability of units to be dropped out
        :param encoder_input: An long tensor with shape of [batch_size, seq_len]
        :param decoder_input: An long tensor with shape of [batch_size, seq_len]
        :param z: An float tensor with shape of [batch_size, latent_variable_size] in case if sampling is performed
        :return: logits for main model and auxiliary logits
                     of probabilities distribution over various tokens in sequence,
                 estimated latent loss
        """

        if z is None:
            [batch_size, _] = encoder_input.size()
            encoder_input = self.embed(encoder_input)
            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = torch.exp(0.5 * logvar)

            z = Variable(torch.randn([batch_size, self.latent_size]))
            if encoder_input.is_cuda:
                z = z.cuda()
            z = z * std + mu
            z = F.dropout(z, drop_prob, training=True)

            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean()
        else:
            kld = None
        decoder_input = self.embed(decoder_input)
        logits, aux_logits = self.decoder(z, decoder_input)

        return logits, aux_logits, kld

    def inference(self, input):

        input = self.embed(input)
        context = self.encoder(input)
        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)

        return mu, logvar

    def sample(self, batch_loader: BatchLoader, use_cuda, z=None):

        if z is None:
            z = Variable(torch.randn([1, self.latent_size]))
            if use_cuda:
                z = z.cuda()

        final_state = None

        cnn_out = self.decoder.conv_decoder(z)

        x = batch_loader.go_input(1, use_cuda)
        x = self.embed(x)

        result = []

        for var in torch.transpose(cnn_out, 0, 1)[:150]:
            out, final_state = self.decoder.rnn_decoder(var.unsqueeze(1), decoder_input=x, initial_state=final_state)
            out = functional.softmax(out.squeeze())

            out = out.data.cpu().numpy()
            idx = batch_loader.sample_char(out)
            x = batch_loader.idx_to_char[idx]

            if x == batch_loader.stop_token:
                break

            result += [x]

            x = Variable(torch.from_numpy(np.array([[idx]]))).long()
            if use_cuda:
                x = x.cuda()
            x = self.embed(x)

        return ''.join(result)
