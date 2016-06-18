import caffe
import numpy as np
from matplotlib import pyplot as plt


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
        if bottom[0].data[:,0,:,:].shape != bottom[1].data[:,0,:,:].shape:
            print "bottom[0].data[:,0,:,:].shape, bottom[1].data[:,0,:,:].shape", bottom[0].data[:,0,:,:].shape, bottom[1].data[:,0,:,:].shape
            raise Exception("Inputs must have the same dimension.")

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)
        
    def forward(self, bottom, top):
        # bottom[0] should be the guess, bottom[1] the image
        # print "sum: ", np.sum(bottom[0].data,axis=1) # Softmax did its job
        
        # This should be used
        answers = bottom[1].data
    
        num_of_bins = bottom[0].data.shape[1] # Conv outputs

        #Is this really useful? Yes, for gaussian dummy input 
        #answers = bottom[1].data - bottom[1].data.min()     # Range [0, ?]
        #answers = answers / answers.max() * 2 - 1 # Range [0, 2], then to [-1, 1]

        # Histogram with uniformly spaced bins
        # input answers are moved from [-1, 1] to [0, 2], then [0, 1], then [0, almost num_of_bins]
        #answers = np.floor((answers + 1) / 2 * (num_of_bins - 0.000001))
        
        [batchsize, _, rowsize, colsize] = bottom[0].data.shape # img, bin, row, col
        cols = np.tile(range(colsize), rowsize)
        rows = np.repeat(range(rowsize), colsize)

        fast_batch_loss = 0
        batch_loss = 0
        for img in range(batchsize):
            img_loss = 0
            # Find the correct bins from the answer
            bins = [answers[img,0,i,j] for i in range(rowsize) for j in range(colsize)] 
            
            # Examine the probability estimates (guesses) of the correct bins
            guesses = bottom[0].data[img,bins,rows,cols] # For all pixels in an image
            
            """
            showguess = guesses.reshape((224,224))
            plt.imshow(showguess)
            plt.colorbar()
            plt.title('Guess')
            
            plt.show()
           
            #cv2.imwrite('/home/ben/image_enhancement/smallnet/visualized_training.png', showguess, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Examine the answer
            plt.subplot(122)
            plt.imshow(answers[0,0,:,:], cmap='gray', interpolation='nearest')
            plt.colorbar()
            plt.title('Answer')
            plt.show()
            """
            # For backpropagation of gradients
            for guess, correct_bin, row, col in zip(guesses, bins, rows, cols):
                if guess == 0:
                    print 'Colorloss.py: Guess was 0 in correct bin'
                    raise Exception('Guess was 0 in correct bin')
                    self.diff[img,correct_bin,row,col] -= 1000 # Would be inf, try 1k instead
                else:
                    self.diff[img,correct_bin,row,col] -= 1/guess # d/dx(-log(x)) = -1/x
                
                if np.isnan(guess):
                    print 'Guess:', guess, 'correct_bin:', correct_bin, 'row: ',  row, 'col: ', col
                    raise Exception('Guess is nan, possibly lr is too large')

                # Add the loss over the whole image
                if np.isnan(fast_batch_loss):
                    raise Exception('Colorloss.py: loss is nan')
                fast_batch_loss -= np.log(guess) # -log(guess) = log(1/guess), KL-divergence

                # if img_loss and batch_loss differ there are numerical problems, and img_loss has 
                # to be added to batch_loss, instead of adding every log(guess) to batch_loss
                img_loss -= np.log(guess)
                
            batch_loss += img_loss
        if batch_loss != fast_batch_loss:
            print batch_loss, fast_batch_loss
            
            print '\nDifference of batch_loss and fast_batch_loss: ', np.sum(batch_loss - fast_batch_loss)
            raise Exception('Numerical error problem, fast_batch_loss cannot be used')

        top[0].data[...] = fast_batch_loss / batchsize
        #print bottom[0].data[0,1,:,:]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff[...]
            #print "gradient of loss: ", bottom[0].diff[...]
            
