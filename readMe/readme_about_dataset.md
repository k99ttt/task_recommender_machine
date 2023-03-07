
## About dataset...

- `X`: This is an (m, $T_x$, 293) dimensional array. 
    - You have m training examples, each of which is a snippet of $T_x = 95 $ works per a day. 
    - At each time step, the input is one of 293 different possible values, represented as a one-hot vector. 
        - For example, X[i,t,:] is a one-hot vector representing the value of the i-th example at time t. 

- `Y`: a $(T_y, m, 293)$ dimensional array
    - This is essentially the same as `X`, but shifted one step to the left (to the past). 
    - Notice that the data in `Y` is **reordered** to be dimension $(T_y, m, 293)$, where $T_y = T_x$. This format makes it more convenient to feed into the LSTM later.
    - Similar to the dinosaur assignment, you're using the previous values to predict the next value.
        - So your sequence model will try to predict $y^{\langle t \rangle}$ given $x^{\langle 1\rangle}, \ldots, x^{\langle t \rangle}$. 

- `n_values`: The number of unique values in this dataset. This should be 293. 

- for converting our corpus to X and Y we can add zeros vector fisrt of our data to build X
and add zeros vector to the end to have Y 