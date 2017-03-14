



main source for params
https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
 For this, we’ll be basing our model off the generator introduced in the DCGAN paper (link: https://arxiv.org/pdf/1511.06434v2.pdf).
 so it is aware of dcgan


1. what is generative modelling
2. what is probability modelling
-implicit/explicit
-gan is implicit
3. what is structured probability modelling
-gan is implicit, strcutured
-why structure is needed and a big table is not enough
4. common example for a gaussian modelling
-like figure 1 in https://arxiv.org/pdf/1701.00160.pdf
-actually this will be the official supplement material
-what is the distribution deteremined by the training data: '1/n'
-what is wat we can approximate with a gaussian
-how a discriminator would ooerate on this?

convolution basics:
 source https://www.tensorflow.org/api_guides/python/nn#Convolution
 http://stackoverflow.com/questions/39373230/what-does-tensorflows-conv2d-transpose-operation-do
 http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed

 conv2d:
  - stride (evaluate this first)
  - padding (evaluate this 2nd, after stride): describe how the stride affects the final image size
  -- 'same': thy phylosopy is that padding may corrupt something but the output size is the same (stride=1) or easy to calculate
  ---- output_size = ceil(input_size/ stride):  examples  ceil(4/2)=2, ceil(3/2)=2, ceil(3/1)=3
  ---- pad(top,bottom,left,right) calculated straightfowardly, @note padding left-right may not symmetric, NO justice!

  -- 'valid': the phylosopy is to "no padding", so padding cannot modify the output, in this sense this is more valid!
  ---- output size = ceil((input_size-filter_size+1)/stride)
  ---- no padding
  ---- @note even if stride= the input image is different: input=3, filter=3, stride=1, output=1

  conv2d weights:
    [filterx]x[filtery] x [in channel] x [out channel]

    example: 2 channel -> 1 channel

        output = filter1 on channel#1 + filter2 on channel#2

    example: 1 channel -> 2 channel

        output[1] = filter1 on channel#1
        output[2] = filter2 on channel#1

    example: 2 channel -> 2 channel

    output[1] = filter1 on channel#1 + filter2 on channel#2
    output[2] = filter1_1 on channel#1 + filter2_2 on channel#2

  conv2d_transpose:
    Transposed convolutions – also called fractionally strided convolutions – work by swapping the forward and backward passes of a convolution.

    -- Convolution as a matrix operation: unroll images, build a big C matrix, where the filter values are repeated.
    --- C matrix is sparse
    --- backpropagation of gradients: C^T can be used

    -stride evaluation: 'think reversed'
    -- this is the best because the formulat is easy: out/stride=input
    --'same':
    --'valid':
    --- example i want a 28x28 image, i want 5x5 kernel, how to desingn an inverse convolution:
    ---- (28-5+1)/2=12 So i need 12x12 as an input image


Backpropagation
   An abbreviation for "backward propagation of errors"
   A common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent.
Gradient descent
   A first-order iterative optimization algorithm.
   Gradient descent is also known as steepest descent, or the method of steepest descent.
   Source: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/