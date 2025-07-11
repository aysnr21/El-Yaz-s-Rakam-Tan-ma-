import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Veri kümesini yükle
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize et
x_train, x_test = x_train / 255.0, x_test / 255.0

# Basit bir model oluştur
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Modeli derle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Eğitim
model.fit(x_train, y_train, epochs=1)  # hızlı olsun diye 1 epoch

# Değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest doğruluğu:', test_acc)

# Tahmin et
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)

# Örnek bir rakamı göster ve tahmin et
index = 0
plt.imshow(x_test[index], cmap=plt.cm.binary)
plt.title(f"Tahmin: {np.argmax(predictions[index])} - Gerçek: {y_test[index]}")
plt.show()
