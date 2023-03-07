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