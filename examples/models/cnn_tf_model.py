from tensorflow import layers, models

def create_cnn_model(
    num_filters_1=32,
    num_filters_2=64,
    kernel_size=3,
    dense_units=512,
    dropout_rate=0.5,
):
    """Builds and returns a configurable CNN model."""
    model = models.Sequential()
    
    model.add(layers.Conv2D(num_filters_1, (kernel_size, kernel_size), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(num_filters_2, (kernel_size, kernel_size), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10)) 

    return model