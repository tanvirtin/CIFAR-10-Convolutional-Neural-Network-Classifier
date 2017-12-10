In order to classify the CIFAR-10 dataset I made two convolutional neural network models, which are named Model-A and Model-B.
```
The architecture of Model-A is as follows:
```
1) Convolutional Layer 1 - 3× 3 ×12 - strides - 1× 1
2) Activated by Relu
3) Max pooling Layer - 2× 2
4) Convolutional Layer 2 - 3× 3 ×24 - strides - 1× 1
5) Activated by Relu
6) Max pooling Layer - 2× 2
7) Convolutional Layer 3 - 3× 3 ×48 - strides - 1× 1
8) Activated by Relu
9) Max pooling Layer - 2× 2
10) Convolutional Layer 4 - 3× 3 ×96 - strides - 1× 1
11) Activated by Relu
12) Max pooling Layer - 2× 2
13) Flatten layer
14) Fully connected layer - 512 neurons
15) Activated by Relu
16) Dropout - 40%
17) Fully connected layer - 512 neurons
18) Activated by Relu
19) Droptout - 40%
20) Output layer - 10 neurons
21) Activated by Softmax

```
The architecture of Model-B is as follows:
```
1) Convolutional Layer 1 - 3× 3 ×24 - strides - 1× 1
2) Activated by Relu
3) Max pooling Layer - 2× 2
4) Convolutional Layer 2 - 3× 3 ×24 - strides - 1× 1
5) Activated by Relu
6) Max pooling Layer - 2× 2
7) Flatten layer
8) Fully connected layer - 512 neurons
9) Activated by Relu
10) Dropout - 25%
11) Fully connected layer - 256 neurons
12) Activated by Relu
13) Droptout - 25%
14) Output layer - 10 neurons
15) Activated by Softmax

After training both the networks the clear observation was that size of the network is important when it comes to accuracy. Model-A with more convo-
lutional layer had a better accuracy than the Model-B with half the convo-lutional layers of Model-A.

The computational graphs, cost vs epochs chart and accuracy vs epochs
chart for both Model-A and Model-B is provided below.
The x-axis of both the graphs is epochs and the y-axis is the accuracy orcost.

![modelacg](https://user-images.githubusercontent.com/25164326/33798870-a0f9921e-dcee-11e7-811c-c347533d9c61.png)
<p align="center">
Computational Graph Model-A
</p>

![modelaacc](https://user-images.githubusercontent.com/25164326/33798871-a2b72134-dcee-11e7-8bd8-536b3ab50dba.PNG)
<p align="center">
Model-A Accuracy vs Epochs
</p>

![modelaloss](https://user-images.githubusercontent.com/25164326/33798872-a746790c-dcee-11e7-8259-2c319b98131f.PNG)
<p align="center">
Model-A Cost vs Epochs
</p>

![modelbcg](https://user-images.githubusercontent.com/25164326/33798874-a9f7492e-dcee-11e7-94d1-c34c4ab2e546.png)
<p align="center">
Computational Graph Model-B
</p>

![modelbacc](https://user-images.githubusercontent.com/25164326/33798876-ac1719be-dcee-11e7-899a-1062e28eb0e1.PNG)
<p align="center">
Model-B Accuracy vs Epochs
</p>

![modelbloss](https://user-images.githubusercontent.com/25164326/33798877-ada07730-dcee-11e7-86f6-c174ae1843cb.PNG)
<p align="center">
Model-B Cost vs Epochs
</p>


