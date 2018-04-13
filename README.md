# implemant DCGAN by Keras

[こちらのコード](https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py)を参考に
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
の論文の実装をしました．

# 実験条件
基本的な実験条件などは論文にかいてあるとおりです．
## モデル
### Generator
```shell:bash
$ python generator
```
### Discriminator
```shell:bash
$ python discriminator
```

# 損失関数
### Generator
<img src="https://latex.codecogs.com/gif.latex?\min&space;\frac{1}{N}&space;\sum_{n=1}^N&space;\log{D\left(G(\boldsymbol{z}_n)\right)}" />

### Discriminator
<img src="https://latex.codecogs.com/gif.latex?\min&space;\frac{1}{N}&space;\sum_{n=1}^N&space;\left\{&space;\log\left[&space;D\left(\boldsymbol{x_n}\right)&space;\right]&space;&plus;&space;\log\left[&space;1-&space;D\left(G(\boldsymbol{z}_n)\right)&space;\right]&space;\right\}" />
