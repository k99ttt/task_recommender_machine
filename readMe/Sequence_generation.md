## Sequence generation uses a for-loop
* If you're building an RNN where, at test time, the entire input sequence $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \ldots, x^{\langle T_x \rangle}$ is given in advance, then Keras has simple built-in functions to build the model. 
* However, for **sequence generation, at test time you won't know all the values of $x^{\langle t\rangle}$ in advance**.
* Instead, you'll generate them one at a time using $x^{\langle t\rangle} = y^{\langle t-1 \rangle}$. 
    * The input at time "t" is the prediction at the previous time step "t-1".
* So you'll need to implement your own for-loop to iterate over the time steps. 
#### Shareable weights
* The function `trmModel()` will call the LSTM layer $T_x$ times using a for-loop.
* It is important that all $T_x$ copies have the same weights. 
    - The $T_x$ steps should have shared weights that aren't re-initialized.
* Referencing a globally defined shared layer will utilize the same layer-object instance at each time step.
* The key steps for implementing layers with shareable weights in Keras are: 
1. Define the layer objects (you'll use global variables for this).
2. Call these objects when propagating the input.

#### 3 types of layers
* The layer objects you need for global variables have been defined.  
    * Just run the next cell to create them! 
* Please read the Keras documentation and understand these layers: 
    - [Reshape()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape): Reshapes an output to a certain shape.
    - [LSTM()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM): Long Short-Term Memory layer
    - [Dense()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense): A regular fully-connected neural network layer.
    
    
### trmModel

Implement `trmModel()`.

#### Inputs (given)

* The `Input()` layer is used for defining the input `X` as well as the initial hidden state 'a0' and cell state `c0`.
* The `shape` parameter takes a tuple that does not include the batch dimension (`m`).
    - For example,
 ```Python
X = Input(shape=(Tx, n_values)) # X has 3 dimensions and not 2: (m, tx, n_values)
```
    
#### Step 1: Outputs

* Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.

#### Step 2: Loop through time steps
* Loop for $t \in 1, \ldots, T_x$:

#### 2A. Select the 't' time-step vector from `X`.
* X has the shape (m, Tx, n_values).
* The shape of the 't' selection should be (n_values,). 
* Recall that if you were implementing in numpy instead of Keras, you would extract a slice from a 3D numpy array like this:
```Python
var1 = array1[:,1,:]
```
    
#### 2B. Reshape `x` to be (1, n_values).
* Use the `reshaper()` layer.  This is a function that takes the previous layer as its input argument.

#### 2C. Run `x` through one step of `LSTM_cell`.

* Initialize the `LSTM_cell` with the previous step's hidden state $a$ and cell state $c$. 
* Use the following formatting:
```python
next_hidden_state, _, next_cell_state = LSTM_cell(inputs=input_x, initial_state=[previous_hidden_state, previous_cell_state])
```
    * Choose appropriate variables for inputs, hidden state and cell state.

#### 2D. Dense layer
* Propagate the LSTM's hidden state through a dense+softmax layer using `densor`. 
    
#### 2E. Append output
* Append the output to the list of "outputs".

#### Step 3: After the loop, create the model
* Use the Keras `Model` object to create a model. There are two ways to instantiate the `Model` object. One is by subclassing, which you won't use here. Instead, you'll use the highly flexible Functional API, which you may remember from an earlier assignment in this course! With the Functional API, you'll start from your Input, then specify the model's forward pass with chained layer calls, and finally create the model from inputs and outputs.

* Specify the inputs and output like so:
```Python
model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
```
    * Then, choose the appropriate variables for the input tensor, hidden state, cell state, and output.
* See the documentation for [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)    