import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import  feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


number_of_features=2
number_of_units=1
weight=tf.Variable(
    tf.zeros([number_of_features,number_of_units])
)
bias=tf.Variable(tf.zeros([number_of_units]))


def perceptron(x):
    i=tf.add(tf.matmul(x,weight),bias)
    output=tf.sigmoid(i)
    return output

individual_loss=lambda : abs(
    tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=perceptron(x))
    )
)



optimizer=tf.keras.optimizers.Adam(.01)
dataframe=pd.read_csv("data.csv")
dataframe.head( )
plt.scatter(dataframe.x1,dataframe.x2,c=dataframe.label)
#plt.show()

x_input=dataframe[['x1','x2']].values
y_label=dataframe[['label']].values

x=tf.Variable(x_input)
x=tf.cast(x,tf.float32)

y=tf.Variable(y_label)
y=tf.cast(y,tf.float32)

for i in range(100):
    with tf.GradientTape() as tape:
        loss=individual_loss()
    gradients=tape.gradient(loss,[weight,bias])    
    optimizer.apply_gradients(zip(gradients,[weight,bias]))
    
tf.print(  weight,bias)

final_loss=tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=perceptron(x))
)
tf.print(final_loss)