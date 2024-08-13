# Deep-Learning CNN Model 🤖📊
## For Image Classification Of Clothing & Accessories 👕👖👟👜

> ## Doployed Project Link: [Streamlit<sup>🔗</sup>](https://image-classification-deep-learning-project-uci44sbugdzmpx5zgoz.streamlit.app/)

### Dataset \:- Fashion MNIST
The Fashion-MNIST clothing classification problem is a new standard dataset used in computer vision and deep learning. It is proposed as a more challenging replacement dataset for the MNIST dataset.

The Fashion MNIST dataset was developed as a response to the wide use of the MNIST dataset, that has been effectively “solved” given the use of modern convolutional neural networks.

Fashion-MNIST was proposed to be a replacement for MNIST, and although it has not been solved, it is possible to routinely achieve error rates of 10% or less. Like MNIST, it can be a useful starting point for developing and practicing a methodology for solving image classification using convolutional neural networks.

It is a dataset comprised of 60,000 small square 28×28 pixel grayscale images of items of 10 types of clothing, such as shoes, t-shirts, dresses, and more. The mapping of all 0-9 integers to class labels is listed below.

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

It is a more challenging classification problem than MNIST and top results are achieved by deep learning convolutional neural networks with a classification accuracy of about 90% to 95% on the hold out test dataset.

![Dataset Images](https://camo.githubusercontent.com/b81b12294aa4a22806429872eafbc0398d09e9e07adf85ade70d68f57efad46b/68747470733a2f2f74656e736f72666c6f772e6f72672f696d616765732f66617368696f6e2d6d6e6973742d7370726974652e706e67)


> ### CNN Model
> ### The model has 2 main aspects : 
> 1. #### The feature extraction front end comprised of convolutional and pooling layers
>       For the convolutional front-end, we can start with a single convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer.
> 
> 2. #### The classifier backend that will make a prediction
>       Given that the problem is a multi-class classification, we know that we will require an output layer with 10 nodes in order to predict the probability distribution of an image belonging to each of the 10 classes. This will also require the use of a softmax activation function. Between the feature extractor and the output layer, we can add a dense layer to interpret the features, in this case with 100 nodes.
>
> - All layers will use the **ReLU** activation function and the He weight initialization scheme, both best practices.
>
> - We will use a conservative configuration for the **stochastic gradient descent** optimizer with a **_learning rate of 0.01_** and a **_momentum of 0.9_**. <br><br>
---
> ### Evalution Of Model :
```
z, acc = model.evaluate(testX, testY, verbose=0)
print('Accuracy : %.3f' % (acc * 100.0))
```
>        Accuracy : 91.140%

| Test Image | Result |
|:----------:|:------:|
| <img src=> | <img src=> |

#### Refernce Links:
- *Basic classification: Classify images of clothing* <https://www.tensorflow.org/tutorials/keras/classification>
- *Deep Learning CNN for Fashion-MNIST Clothing Classification* <https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/>
- *tensorflow Github Repo* <https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb>
