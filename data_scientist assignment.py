# Fine tuning scripit for emotion detection 


import tensorflow as tf
base_model = tf.keras.applications.xception.Xception(weights="imagenet",include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg) #n_classes: number of different emotions
model = tf.keras.Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers[56:]:
    layer.trainable = True  #we can hypertune the parameter 56
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) #implemented optimisers and corresponding learning rate
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
 metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=1000,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)])
#epochs is high because we are using early stopping
model.save("emotion_detection.h5")