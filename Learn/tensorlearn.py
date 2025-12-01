
#gradient tape
# w = tf.Variable(4.0)
# b = tf.Variable(2.0)

# with tf.GradientTape() as tape:
#     y = w * 3 + b

# # Compute gradient of y w.r.t. [w, b]
# grads = tape.gradient(y, [w, b])
# print("dy/dw:", grads[0].numpy())  # 3
# print("dy/db:", grads[1].numpy())  # 1



#single neural network


# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense

# # Step 1: Input Layer
# inputs = Input(shape=(3,))  # 3 features

# # Step 2: Hidden Layer
# x = Dense(64, activation='relu')(inputs)

# # Step 3: Output Layer
# outputs = Dense(1, activation='sigmoid')(x)

# # Step 4: Wrap into a Model
# model = Model(inputs=inputs, outputs=outputs)



