import torch 
import torch.nn as nn
from shared.utils import batch_generator


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, layers):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers = layers, batch_first = True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = layers, bidirectional=True, batch_first = True)
        self.linear = nn.Linear(hidden_dim*2, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x) 
        return x


def wrgan_gp(ori_data, params):
    generator = Generator(params.latent_dim, params.input_size, params.num_layers).to(params.device)
    discriminator = Discriminator(params.input_size, params.hidden_size, params.num_layers).to(params.device)

    
    # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
    gen_optim = torch.optim.Adam(generator.parameters(), lr= 0.001, betas=(0.5, 0.9))
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr= 0.001, betas=(0.5, 0.9))
    
    # Batch generator, it keeps on generating batches of data.
    data_gen = batch_generator(ori_data, params)
    with torch.backends.cudnn.flags(enabled=False):

        for step in range(params.max_steps):
            for disc_step in range(params.disc_extra_steps):
                """
                Discriminator training.
                
                - Generate fake data from the generator.
                - Train the discriminator on the real data and the fake data.

                Note: Make sure to detach the variable from the graph to prevent backpropagation. 
                    in this case, it is the synthetic data, (generator(noise)).
                """
                # Get the real batch data, and synthetic batch data. 
                bdata = data_gen.__next__() 
                noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)

                fake = generator(noise).detach() 
                fake_dscore = discriminator(fake)
                true_dscore = discriminator(bdata)

                # Compute gradient penalty, as per the paper.
                epsilon = torch.rand(params.batch_size, 1, 1).to(params.device)
                # x_hat represents and interpolated sample between real and fake data.
                x_hat = (epsilon * bdata + (1 - epsilon) * fake).requires_grad_(True)  
                dscore_hat = discriminator(x_hat)
                gradients = torch.autograd.grad(outputs=dscore_hat, inputs=x_hat, grad_outputs=torch.ones(dscore_hat.size()).to(params.device), create_graph=True, retain_graph=True, only_inputs=True)[0]

                # Compute the penalty.
                gp = torch.sqrt(torch.sum(gradients** 2, dim=1) + 1e-10)
                gp = torch.mean((gp-1)**2)

                # Compute the loss.
                wasserstein_distance = torch.mean(true_dscore) - torch.mean(fake_dscore)
                penalty = params.gp_lambda * gp
                dloss = -wasserstein_distance + penalty
                disc_optim.zero_grad()
                dloss.backward()
                disc_optim.step()
            
            """Generator training."""
            # Generate fake data from the generator.
            noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)
            fake = generator(noise) 
            fake_dscore = discriminator(fake)

            # Compute the loss for the generator, and backpropagate the gradients.
            gloss = -1.0*torch.mean(fake_dscore)
            gen_optim.zero_grad()
            gloss.backward()
            gen_optim.step()
            print('[Step {}; L(C): {}; L(G): {}; Wass_Distance: {}]'.format(step + 1, dloss, gloss, wasserstein_distance))



    noise = torch.randn(ori_data.shape[0], params.seq_len, params.latent_dim).to(params.device)
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples



