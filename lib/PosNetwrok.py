import torch
import torch.nn as nn
import torch.nn.functional as F

# class PositionalEncoding(nn.Module):
#     def __init__(self, input_dim, num_frequencies):
#         """
#         Positional Encoding module to map input coordinates to a higher dimensional space.
#         Args:
#             input_dim (int): Dimensionality of the input (e.g., 3 for (x, y, z)).
#             num_frequencies (int): Number of frequency bands for encoding.
#         """
#         super(PositionalEncoding, self).__init__()
#         self.num_frequencies = num_frequencies
#         self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

#     def forward(self, x):
#         """
#         Encode input coordinates with sinusoidal functions.
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, N, input_dim).
#         Returns:
#             torch.Tensor: Encoded tensor of shape (B, N, input_dim * 2 * num_frequencies).
#         """
#         # Apply positional encoding
#         x_expanded = x.unsqueeze(-1) * self.freq_bands.to(x.device)  # (B, N, input_dim, num_frequencies)
#         x_encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)  # (B, N, input_dim, 2*num_frequencies)
#         return x_encoded.view(x.shape[0], x.shape[1], -1)  # (B, N, input_dim * 2 * num_frequencies)


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        """
        Positional Encoding module to map input coordinates to a higher dimensional space.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 3 for (x, y, z)).
            num_frequencies (int): Number of frequency bands for encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        print(self.freq_bands)
        print(self.freq_bands.shape)
    def forward(self, x):
        """
        Encode input coordinates with sinusoidal functions.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, N, input_dim * 2 * num_frequencies).
        """
        # Apply positional encoding
        x_expanded = x.unsqueeze(-1) * self.freq_bands.to(x.device)  # (B, N, input_dim, num_frequencies)
        x_encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)  # (B, N, input_dim, 2*num_frequencies)
        return x_encoded.view(x.shape[0], x.shape[1], -1)  # (B, N, input_dim * 2 * num_frequencies)




class DeformableNetworkWithPE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3, num_layers=4, num_frequencies=20,zero_inited=False):
        """
        Deformable Network with Positional Encoding.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 3 for (x, y, z)).
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Dimensionality of output (e.g., 3 for deformation offsets).
            num_layers (int): Number of layers in the MLP.
            num_frequencies (int): Number of frequencies for positional encoding.
        """
        super(DeformableNetworkWithPE, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = NerfPositionalEncoding(input_dim, num_frequencies)
        encoded_dim = input_dim * 2 * num_frequencies

        # layer_norm = nn.LayerNorm(embedding_dim)
        # # Activate module
        # layer_norm(embedding)

        # MLP
        layers = []
        for i in range(num_layers):
            in_dim = encoded_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
           
            if i < num_layers - 1:
                layers.append(nn.Linear(in_dim, out_dim,bias=False))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_dim, out_dim,bias=False))
                # if i ==1:
                #     layers.append(nn.LayerNorm(out_dim))
            # else:
            #     layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                1
                # Xavier Uniform Initialization for weights
                # nn.init.sparse_(layer.weight,sparsity=0.3)
                # Zero initialization for biases
            # if hasattr(layer, 'bias'):
            #     if layer.bias!=None:
            #         nn.init.uniform_(layer.bias,0.1/128.0)
        
        if zero_inited:
            nn.init.zeros_(self.mlp[-1].weight)
            #nn.init.zeros_(self.mlp[-1].bias)

        self.last_act = nn.Tanh()
        
    def forward(self, x):
        """
        Forward pass for deformable network with positional encoding.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Deformed offsets of shape (B, N, output_dim).
        """
        # Apply positional encoding
        x_encoded = self.positional_encoding(x)

        # Predict deformation offsets
        mlped =  self.mlp(x_encoded)
        # print(mlped.shape)
        # print(mlped.mean(-2).shape)
        # return mlped
        # with torch.no_grad():
        # center = mlped.mean(-2)
        # return mlped-center
    #-mlped.mean(-2)*1.0
        # return mlped
        return mlped
        import random
        probability = 0.3
        if random.random() < probability:
            return mlped-mlped.mean(-2)*1.0
        else:
            return mlped
        return self.last_act(mlped)
    
    
import torch
import torch.nn as nn
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        """
        Learnable Positional Encoding for high-resolution fitting.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 2 for (x, y)).
            num_frequencies (int): Number of frequency bands for encoding.
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies

        # Learnable frequency bands (shared across dimensions)
        self.freq_bands = nn.Parameter(torch.randn(num_frequencies) * 3.14)

    def forward(self, x):
        """
        Encode input coordinates with sinusoidal functions.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, N, input_dim * 2 * num_frequencies).
        """
        # (B, N, input_dim) -> (B, N, input_dim, num_frequencies)
        x_expanded = x.unsqueeze(-1) * self.freq_bands  # Broadcast frequencies to all dimensions
        x_encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)  # Sin and Cos
        return x_encoded.view(x.shape[0], x.shape[1], -1)  # Flatten to (B, N, input_dim * 2 * num_frequencies)


class ImprovedDeformableNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=8, num_frequencies=20):
        """
        Improved Deformable Network with Learnable Positional Encoding and Residual Connections.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 2 for (x, y)).
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Dimensionality of output (e.g., 3 for RGB values).
            num_layers (int): Number of layers in the MLP.
            num_frequencies (int): Number of frequencies for positional encoding.
        """
        super(ImprovedDeformableNetwork, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = LearnablePositionalEncoding(input_dim, num_frequencies)
        encoded_dim = input_dim * 2 * num_frequencies

        # MLP
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = encoded_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
        
        # Activation functions
        self.activation = nn.ReLU()
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass for improved deformable network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, output_dim).
        """
        # Apply positional encoding
        x_encoded = self.positional_encoding(x)

        # MLP forward pass
        h = x_encoded
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        
        return self.final_activation(h)


class NerfPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        """
        NeRF-style Positional Encoding.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 2 for (x, y)).
            num_frequencies (int): Number of frequency bands for encoding.
        """
        super(NerfPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)  # [1, 2, 4, ..., 2^(num_frequencies-1)]
        self.freq_bands = self.freq_bands.to('cuda')  # Move freq_bands to the same device as x
        # print(self.freq_bands)
        # print(self.freq_bands.shape)
        # self.freq_bands[0] = 0.0
        # self.freq_bands[1] = 0.0
    def forward(self, x):
        """
        Encode input coordinates with sinusoidal functions.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, N, input_dim * 2 * num_frequencies).
        """
        
        x_expanded = torch.einsum("bni,j->bnij", x, self.freq_bands)  # (B, N, input_dim) -> (B, N, input_dim, num_frequencies)
        x_encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)  # (B, N, input_dim, 2*num_frequencies)
        return x_encoded.view(x.shape[0], x.shape[1], -1)  # Flatten to (B, N, input_dim * 2 * num_frequencies)


class NerfMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=8, num_frequencies=10):
        """
        MLP with NeRF-style Positional Encoding.
        Args:
            input_dim (int): Dimensionality of the input (e.g., 2 for (x, y)).
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Dimensionality of output (e.g., 3 for RGB values).
            num_layers (int): Number of layers in the MLP.
            num_frequencies (int): Number of frequency bands for positional encoding.
        """
        super(NerfMLP, self).__init__()
        
        # NeRF Positional Encoding
        self.positional_encoding = NerfPositionalEncoding(input_dim, num_frequencies)
        encoded_dim = input_dim * 2 * num_frequencies

        # MLP
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = encoded_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
        
        # Activation functions
        self.activation = nn.ReLU()
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass for NeRF MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, output_dim).
        """
        # Apply positional encoding
        x_encoded = self.positional_encoding(x)

        # MLP forward pass
        h = x_encoded
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        
        return self.final_activation(h)/100.0
