import tensorflow as tf


mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train, x_test=x_train / 255.0, x_test / 255.0

digit = tf.keras.models.Sequential(#Arquitectura de la red neuronal
    [
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax),#Capa densa de 10 neuronas cuya funcion de activación es softmax
    ]
)

digit.compile(#Compilar modelo
    optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]
)

digit.fit(x_train,y_train,epochs=3)
digit.evaluate(x_test,y_test)
