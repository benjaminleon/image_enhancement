import caffe
import numpy as np

class ColorLossLayer(caffe.Layer):
    """
    Compute the loss between a probability estimate for a or b color in ab-space
    and the one-hot map of what the color actually is. For all pixels in an 
    input image
    """
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        #if bottom[0].data[:,0,:,:] != bottom[1].data[:,0,:,]:
        #    raise Exception("Inputs must have the same dimension.")
        
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)
        
    def forward(self, bottom, top):
        # bottom[0] should be the guess, bottom[1] the image
        # print "sum: ", np.sum(bottom[0].data,axis=1) # Softmax did its job
        
        # This should be used
        # answer_image = bottom[1].data
        # Images has values in [-1, 1]
        answer_image = bottom[1].data - bottom[1].data.min()     # Range [0, ?]
        answer_image = answer_image / answer_image.max() * 2 - 1 # Range [0, 2], then [-1, 1]
        
        num_of_bins = bottom[0].data.shape[1] # Conv outputs
        
        # Histogram with uniformly spaced bins
        # input is moved from [-1, 1] to [0, 2], then [0, 1], then [0, almost num_of_bins]
        answers = np.floor((answer_image + 1) / 2 * (num_of_bins - 0.000001))
        
        [batchsize, _, rowsize, colsize] = bottom[0].data.shape # img, bin, row, col
        cols = np.tile(range(colsize), rowsize)
        rows = np.repeat(range(rowsize), colsize)

        batch_loss = 0
        for img in range(batchsize):
            # Find the correct bins from the answer
            bins = [answers[img,0,i,j] for i in range(rowsize) for j in range(colsize)] 

            # Examine the probability estimates (guesses) of the correct bins
            guesses = bottom[0].data[img,bins,rows,cols] # For all pixels in an image
            
            # For backpropagation of gradients
            for guess, correct_bin, row, col in zip(guesses, bins, rows, cols):
                self.diff[img,correct_bin,row,col] -= 1/guess # d/dx(-log(x)) = -1/x
                
                # Add the loss over the whole image
                batch_loss += -np.sum(np.log(guess)) # -log(guess) = log(1/guess), as in KL-divergence
        
        top[0].data[...] = batch_loss / batchsize
        print "The loss was: ", top[0].data
    
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff[...]
            #print "gradient of loss: ", bottom[0].diff[...]
            
