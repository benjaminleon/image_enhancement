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
        print "guesses", bottom[0].data.shape
        print bottom[0].data
        # print "sum: ", np.sum(bottom[0].data,axis=1) # Softmax did its job
        
        # Synthetic answer. The values represent the correct bin
        # TODO: Make answer from input image
        # Perhaps a new layer which converts images to indices of correct bins
        print "answer: ", bottom[1].data.shape
        bottom[1].data[0,0,0,0] = 0 # img, bin, row, col
        bottom[1].data[0,0,0,1] = 1
        bottom[1].data[0,0,0,2] = 1
        bottom[1].data[0,0,1,0] = 0
        bottom[1].data[0,0,1,1] = 0
        bottom[1].data[0,0,1,2] = 1

        bottom[1].data[1,0,0,0] = 0
        bottom[1].data[1,0,0,1] = 0
        bottom[1].data[1,0,0,2] = 1
        bottom[1].data[1,0,1,0] = 1
        bottom[1].data[1,0,1,1] = 0
        bottom[1].data[1,0,1,2] = 0
        print bottom[1].data
        
        [batchsize, _, rowsize, colsize] = bottom[0].data.shape # img, bin, row, col
    
        cols = np.tile(range(colsize), rowsize)
        rows = np.repeat(range(rowsize), colsize)
        print "cols", cols, "rows", rows

        self.diff = 0 # Prepare for gradient backpropagation
        batch_loss = 0
        for img in range(batchsize):
            # Find the correct bins from the answer
            bins = [bottom[1].data[img,0,i,j] for i in range(rowsize) for j in range(colsize)] 

            # Examine the probability estimates (guesses) of the correct bins
            guesses = bottom[0].data[img,bins,rows,cols] # For all pixels in an image
            print "guesses", guesses

            # For backpropagation of gradients
            for guess in guesses:
                print "guess", guess
                self.diff -= 1/guess # d/dx(-log(x)) = -1/x

            # Add the loss over the whole image
            batch_loss += -np.sum(np.log(guess)) # -log(guess) = log(1/guess), as in KL-divergence
            top[0].data[...] = batch_loss
            
    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            bottom[1].diff[...] = self.diff
            
