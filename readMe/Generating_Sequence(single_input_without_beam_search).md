## Generating Sequence(single input without beam search)

You now have a trained model which has learned the patterns of a user tasks. You can now use this model to synthesize new Sequence! 

<a name='3-1'></a>
### 3.1 - Predicting & Sampling
![](music_gen.png)

#### At each step of sampling, you will:
* Take as input the activation '`a`' and cell state '`c`' from the previous state of the LSTM.
* Forward propagate by one step.
* Get a new output activation, as well as cell state. 
* The new activation '`a`' can then be used to generate the output using the fully connected layer, `densor`. 

#### Initialization
* You'll initialize the following to be zeros:
    * `x0` 
    * hidden state `a0` 
    * cell state `c0` 
    
```python

'''
Why should we use 
 - tf.math.argmax()
 - tf.one_hot()
 - RepeatVector() ??

temp = tf.math.argmax([[4,40,4],[4,40,4]],-1)
print(temp)
temp = tf.one_hot(temp,5)
print(temp)
RepeatVector(1)(temp)

tf.Tensor([1 1], shape=(2,), dtype=int64)
tf.Tensor(
[[0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0.]], shape=(2, 5), dtype=float32)
<tf.Tensor: shape=(2, 1, 5), dtype=float32, numpy=
array([[[0., 1., 0., 0., 0.]],

       [[0., 1., 0., 0., 0.]]], dtype=float32)>'''

```