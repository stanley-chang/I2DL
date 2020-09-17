import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nr_layers = 1

        self.W = nn.Linear(hidden_size,hidden_size)
        self.V = nn.Linear(input_size,hidden_size)
        if activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = nn.functional.relu
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        if not h:
            h = torch.zeros(self.nr_layers,x.shape[1],self.hidden_size)
        for x_t in x:
            h = self.activation(self.W(h) + self.V(x_t))
            h_seq.append(h)
        h_seq = torch.cat(h_seq,0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nr_layers = 1

        self.Wf = nn.Linear(input_size,hidden_size)
        self.Wi = nn.Linear(input_size,hidden_size)
        self.Wo = nn.Linear(input_size,hidden_size)
        self.Wc = nn.Linear(input_size,hidden_size)

        self.Uf = nn.Linear(hidden_size,hidden_size)
        self.Ui = nn.Linear(hidden_size,hidden_size)
        self.Uo = nn.Linear(hidden_size,hidden_size)
        self.Uc = nn.Linear(hidden_size,hidden_size)

        self.sigma_g = torch.sigmoid
        self.sigma_c = torch.tanh
        self.sigma_h = torch.tanh
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []


        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        if not h:
            h = torch.zeros(self.nr_layers,x.shape[1],self.hidden_size)
        if not c:
            c = torch.zeros(self.nr_layers,x.shape[1],self.hidden_size)

        for x_t in x:
            f_t = self.sigma_g(self.Wf(x_t) + self.Uf(h))
            i_t = self.sigma_g(self.Wi(x_t) + self.Ui(h))
            o_t = self.sigma_g(self.Wo(x_t) + self.Uo(h))
            # print(f_t.shape, c.shape)
            c = f_t * c  + i_t * self.sigma_c(self.Wc(x_t) + self.Uc(h))
            h = o_t * self.sigma_h(c)
            h_seq.append(h)
        h_seq = torch.cat(h_seq,0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq , (h, c)

