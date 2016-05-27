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
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

        def forward(self, bottom, top):
            # First version of this function only works for 1 pixel.
            # bottom[0] should be the image, bottom[1] the guess
            self.diff[...] = np.log(bottom[1].data[np.argmax(bottom[0].data[...])])
            top[0].data[...] = self.diff
            
        def backward(self, top, propagate_down, bottom):
            if not propagate_down[1]:
                continue
            bottom[1].diff[...] = 1/bottom[1].data[np.argmax(bottom[0].data[...])]
                    
