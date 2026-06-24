import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
labels = []

dataset_path = "dataset_g"

for gesture in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, gesture)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        arr = np.load(path)

        # ensure correct size
        if len(arr) < 126:
            arr = np.pad(arr, (0,126-len(arr)))

        if len(arr) > 126:
            arr = arr[:126]

        data.append(arr)
        labels.append(gesture)

data = np.array(data)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

labels = to_categorical(labels)

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(126,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(labels.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

X_train, X_test, y_train, y_test = train_test_split(
data, labels, test_size=0.2, random_state=42
)
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
os.makedirs("model", exist_ok=True)

model.save("model/gesture_model.h5")

np.save("model/labels.npy", encoder.classes_)

print("Training complete")