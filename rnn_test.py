
class RNN(object):
    def __init__(self, input_units, output_units, hidden_units):
        self.W_in = theano.shared()
        self.W_h = theano.shared
        self.W_out = theano.shared()
        self._activation = T.tanh
        self.h0 = T.vector()

    def compute(self, h, x):
        h_next = T.tanh(T.dot(self.W_in, x) + T.dot(self.W_h, h))
        y_next = T.dot(self.W_out, h_next)
        return h_next, y_next


    def forward(self, X, W_in, W_h, W_out):
        return T.scan(fn = self.compute, 
                        outputs_info = None, 
                        sequences = [x], 
                        nonsequences = [W_in, W_h, W_out])     


    def train(self, X, learning_rate = 1e-3):
        
        def cost(X, W_in, W_h, W_out):
            H, Y = self.forward(X, W_in, W_h, W_out)
            return ((Y - X)**2).sum()

        gW_in, gW_h, gW_out = T.grad(self.cost, [W_in, W_h, W_out])
        calculate = theano.function(inputs=[X], 
                                    outputs=cost, 
                                    updates = 
                                        {   W_in: W_in - learning_rate * gW_in,
                                            W_h: W_h - learning_rate * gW_h,
                                            W_out: W_out - learning_rate * gW_out
                                        }
                                    )



