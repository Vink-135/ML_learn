# (To see the working of layers see this :https://colab.research.google.com/drive/1HmL5auiKu3vbKDOTjbnofEmsqMWYViG9?usp=sharing)

# 🧪 Let’s Code: Transfer Learning with VGG16 (Simple Example)

# ✅ Step 1: Load Your Dataset (We’ll use CIFAR-10 or custom folder)

# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Normalize
# X_train, X_test = X_train/255.0, X_test/255.0

# # One-hot encode labels
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)



# ✅ Step 2: Load VGG16 without top layer

# from tensorflow.keras.applications import VGG16

# base_model = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
# base_model.trainable = False  # Freeze all layers

# ✅ Step 3: Add Your Own Classifier on Top

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout

# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')  # CIFAR-10 has 10 classes
# ])


# ✅ Step 4: Compile & Train

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


# ✅ Step 5: Evaluate

# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {acc:.4f}")


# 📊 Bonus: Fine-Tune the Top Layers

# base_model.trainable = True

# # Fine-tune only the top N layers
# for layer in base_model.layers[:-4]:
#     layer.trainable = False

# # Recompile and retrain
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)




# ------------------------------------------------------------------------------------------------------------------------------------------------



# ✅ Fine-Tuning a Pretrained CNN (ResNet50 on CIFAR-10)


# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# # Load data
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Normalize
# X_train, X_test = X_train / 255.0, X_test / 255.0  #so that our data ranges from 0 to 1

# # One-hot encode
# y_train = to_categorical(y_train, 10) #similarly this is done as there are 10 classes in cifar that to conevrting them into 0 and 1 
# y_test = to_categorical(y_test, 10)  #matching class 1 others will be 0


# 🔹 Step 2: Load Pretrained Base Model (without top)

# from tensorflow.keras.applications import ResNet50

# base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

# # Freeze the base model initially (for feature extraction)
# base_model.trainable = False


# 🔹 Step 3: Add Custom Classifier on Top

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(10, activation='softmax')
# ])


# 🔹 Step 4: Compile & Train (Feature Extraction Phase)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)


# 🔹 Step 5: Fine-Tune the Top Layers of Base Model


# # Unfreeze the base model
# base_model.trainable = True

# # Optional: Freeze lower layers, unfreeze only last few layers
# for layer in base_model.layers[:-30]:
#     layer.trainable = False


# 🔹 Step 6: Recompile with Lower Learning Rate

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 🔹 Step 7: Evaluate Final Model

# loss, acc = model.evaluate(X_test, y_test)
# print(f"✅ Fine-tuned Model Accuracy: {acc:.4f}")




# If you want to see which layers are frozen/trainable, you can print:

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name, layer.trainable)




# <------------------------------------------------------------------------------------------------------------------------------------------------>


# RESNET



# 🧪 Build a Simple ResNet Block (Code) (NOT VERY IMPORTANT)

# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
# from tensorflow.keras import Input, Model

# def resnet_block(x, filters):
#     shortcut = x  # skip connection

#     x = Conv2D(filters, kernel_size=3, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = Conv2D(filters, kernel_size=3, padding='same')(x)
#     x = BatchNormalization()(x)

#     # Add skip connection
#     x = Add()([x, shortcut])
#     x = Activation('relu')(x)

#     return x


# 🔧 Use Pretrained ResNet (Fast Method)

# from tensorflow.keras.applications import ResNet50

# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# resnet.trainable = False



# <------------------------------------------------------------------------------------------------------------------------------------------------>


#U-NET

# 🔹 Step 1: U-Net Block (Basic)

# from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, UpSampling2D
# from tensorflow.keras.models import Model

# def unet_model(input_shape=(128, 128, 1)):
#     inputs = Input(input_shape)

#     # Encoder
#     c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D(2)(c1)

#     c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
#     c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
#     p2 = MaxPooling2D(2)(c2)

#     # Bottleneck
#     c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
#     c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)

#     # Decoder
#     u1 = UpSampling2D(2)(c3)
#     u1 = concatenate([u1, c2])
#     c4 = Conv2D(128, 3, activation='relu', padding='same')(u1)
#     c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)

#     u2 = UpSampling2D(2)(c4)
#     u2 = concatenate([u2, c1])
#     c5 = Conv2D(64, 3, activation='relu', padding='same')(u2)
#     c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)

#     # Output
#     outputs = Conv2D(1, 1, activation='sigmoid')(c5)

#     return Model(inputs, outputs)




# 🧪 How to Use It

# Image → Model → Mask



# model = unet_model()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # X = image input (128x128x1)
# # Y = pixel-wise masks (128x128x1)
# model.fit(X, Y, epochs=10)



# <------------------------------------------------------------------------------------------------------------------------------------------------>


#SIAMESE NETWORK


# 🔹 Step 1: Define the Shared Base Network (e.g., Mini CNN)


# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# def build_base_model(input_shape=(100, 100, 1)):
#     input = Input(input_shape)
#     x = Conv2D(32, 3, activation='relu')(input)
#     x = MaxPooling2D()(x)
#     x = Conv2D(64, 3, activation='relu')(x)
#     x = MaxPooling2D()(x)
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     return Model(input, x)

# # 🔹 Step 2: Define Siamese Model

# from tensorflow.keras.layers import Lambda
# import tensorflow.keras.backend as K
# import tensorflow as tf

# def euclidean_distance(vectors):
#     x, y = vectors
#     return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# input_shape = (100, 100, 1)

# # Create inputs
# input_a = Input(input_shape)
# input_b = Input(input_shape)

# # Shared feature extractor
# base_model = build_base_model(input_shape)
# feat_a = base_model(input_a)
# feat_b = base_model(input_b)

# # Distance
# distance = Lambda(euclidean_distance)([feat_a, feat_b])

# # Output (probability it's same)
# output = Dense(1, activation='sigmoid')(distance)

# siamese_model = Model(inputs=[input_a, input_b], outputs=output)


# # 🔹 Step 3: Compile and Train

# siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # X1, X2 = image pairs
# # y = 1 if same, 0 if different
# siamese_model.fit([X1, X2], y, epochs=10)



# <------------------------------------------------------------------------------------------------------------------------------------------------>

#VCG-16/19

# 🔧 Using VGG16 in Keras (Pretrained)

# from tensorflow.keras.applications import VGG16

# model = VGG16(include_top=True, weights='imagenet')  # with FC layers
# model.summary()



# 🔹 Want to Use it for Your Own Task?


# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout

# base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
# base_model.trainable = False  # Freeze weights

# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # for binary classification
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()



# 📦 Preprocessing for VGG


# from tensorflow.keras.applications.vgg16 import preprocess_input

# # x must be 224x224x3
# x = preprocess_input(x)




# <------------------------------------------------------------------------------------------------------------------------------------------------># BERT Model

# (https://colab.research.google.com/drive/1xyaAMav_gTo_KvpHrO05zWFhmUaILfEd?usp=sharing#scrollTo=UKZAgDk9Jnrx)

# above is notebook for understabding the code of bert model

