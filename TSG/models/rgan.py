import torch 
import torch.nn as nn
from shared.utils import batch_generator


class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim, layers):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, input_dim, num_layers = layers, batch_first = True)
        self.linear = nn.Linear(input_dim, input_dim)
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



def rgan(ori_data, params):
    generator = Generator(params.latent_dim, params.input_size, params.num_layers).to(params.device)
    discriminator = Discriminator(params.input_size, params.hidden_size, params.num_layers).to(params.device)

    
    # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
    gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    disc_optim = torch.optim.SGD(discriminator.parameters(), lr = 0.1)

    # Batch generator, it keeps on generating batches of data.
    data_gen = batch_generator(ori_data, params)
    # BCE with logits
    criterion = torch.nn.BCEWithLogitsLoss()

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

            # Compute the loss for the discriminator, and backpropagate the gradients.
            dloss = criterion(fake_dscore, torch.zeros_like(fake_dscore)) + criterion(true_dscore, torch.ones_like(true_dscore))
            disc_optim.zero_grad()
            dloss.backward()
            disc_optim.step()

        noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)
        fake = generator(noise) 
        fake_dscore = discriminator(fake)

        # Compute the loss for the generator, and backpropagate the gradients.
        gloss = criterion(fake_dscore, torch.ones_like(fake_dscore))

        gen_optim.zero_grad()
        gloss.backward()
        gen_optim.step()
        
        print('[Step {}; L(G): {}; L(D): {}]'.format(step, dloss, gloss))



    noise = torch.randn(ori_data.shape[0], params.seq_len, params.latent_dim).to(params.device)
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples



