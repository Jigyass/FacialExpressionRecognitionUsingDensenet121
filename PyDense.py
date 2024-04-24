import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [224, 224], method='bilinear')
    image = tf.image.grayscale_to_rgb(image)
    return image, label

def prepare_dataset(directory, batch_size=32):
    dataset = image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True
    )
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def prepare_datasets(directory, batch_size=100, train_val_split=0.8):
    dataset = image_dataset_from_directory(
        directory,
        validation_split=1-train_val_split,
        subset="training",
        seed=123,
        labels='inferred',
        label_mode='categorical',
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataset = image_dataset_from_directory(
        directory,
        validation_split=1-train_val_split,
        subset="validation",
        seed=123,
        labels='inferred',
        label_mode='categorical',
        image_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True
    )

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, validation_dataset

def build_model(num_classes):
    input_tensor = Input(shape=(224, 224, 3))
    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

if __name__ == "__main__":
    train_directory = '/home/jigyas/Github/FacialExpressionRecognitionUsingDensenet121/project_1_dataset/train'
    train_dataset = prepare_dataset(train_directory)
    model = build_model(7)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_dataset, validation_dataset = prepare_datasets(train_directory)
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10
    )

    DenseCNN = '/home/jigyas/Github/FacialExpressionRecognitionUsingDensenet121/NewDenseCNN'
    Densehd = '/home/jigyas/Github/FacialExpressionRecognitionUsingDensenet121/NewDense.h5'
    DenseKeras = '/home/jigyas/Github/FacialExpressionRecognitionUsingDensenet121/NewDenseKeras.keras'
    model.save(DenseCNN)
    # model.save(Densehd, save_format='h5')
    model.save(DenseKeras, save_format='keras')

