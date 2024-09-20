import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import gc
#from vae import VAE
#import tensorflow_io as tfio
from tensorflow.keras.models import load_model
from absl import logging
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Desativa logs de nível INFO e WARNING


# Gerenciamento da memória GPU conforme necessário
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Define os diretorios
normal_dir = 'D:/POS/auto_dasr/dataset/control/wavs/original'
dysarthric_dir = 'D:/POS/auto_dasr/dataset/dysarthric/wavs/aug_prop'
test_dir = 'D:/POS/auto_dasr/dataset/dysarthric/wavs/original/F05'
reconstruct_base_dir = 'D:/POS/auto_dasr/dataset/reconstruct/hubert'
spectrogram_dir = os.path.join(reconstruct_base_dir, 'spectrograms')
progress_dir = os.path.join(spectrogram_dir, 'progress')
wav_dir = os.path.join(reconstruct_base_dir, 'wav')

# Create directories if they don't exist
os.makedirs(spectrogram_dir, exist_ok=True)
os.makedirs(wav_dir, exist_ok=True)
os.makedirs(progress_dir, exist_ok=True)

BATCH = 16
SIZE = 256  
EPOCHS = 200
RATE_DB = 1
#spec_split = 1
n_fft = 2048
hop_length = 256

def load_wav(file_path):
    audio, _ = sf.read(file_path)
    audio = librosa.util.normalize(audio)
    
    stfts = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    x = np.abs(stfts)
    
    # Normalize values between 0 and 1
    x = x / np.max(x)

    # Resize spectrogram
    x = tf.image.resize(x[..., np.newaxis], (SIZE, SIZE)).numpy()
    x = np.repeat(x, 3, axis=-1)  # Convert to 3 channels

    return x
'''
def save_spectrogram(spectrogram, file_path):
    spectrogram = np.clip(spectrogram, 0, 1)  # Limita os valores entre 0 e 1

    # Converte para decibéis para visualização (apenas no canal 0 para simplificar)
    db_spectrogram = librosa.amplitude_to_db(spectrogram[..., 0], ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(db_spectrogram, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(file_path)
    plt.close()
'''

def save_spectrogram(original, reconstructed, file_path):
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)

    db_original = librosa.amplitude_to_db(original[..., 0], ref=np.max)
    db_reconstructed = librosa.amplitude_to_db(reconstructed[..., 0], ref=np.max)

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(db_original, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(db_reconstructed, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed Spectrogram')

    plt.savefig(file_path)
    plt.close()
    
def plot_spectrograms(input_spec, target_spec, output_spec, epoch, batch):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, spec, title in zip(axes, [input_spec, target_spec, output_spec], ['Input', 'Target', 'Output']):
        spec_db = librosa.amplitude_to_db(spec[..., 0], ref=np.max)
        img = librosa.display.specshow(spec_db, sr=16000, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title)

    filename = f'epoch_{epoch}_batch_{batch}.png'
    filepath = os.path.join(progress_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f'File {filename} saved!')

def audio_to_spectrogram(audio):
    spectrogram = librosa.stft(audio)
    return np.abs(spectrogram)

def spectrogram_to_audio(spectrogram):
    audio = librosa.istft(spectrogram)
    return audio

wavs_normal = glob("{}/**/*.wav".format(normal_dir), recursive=True)
wavs_dys = glob("{}/**/*.wav".format(dysarthric_dir), recursive=True)
wavs_test = glob("{}/**/*.wav".format(test_dir), recursive=True)

# Extract base names (without '_aug.wav')
def get_base_name(file_list, suffix='_aug.wav'):
    return {os.path.basename(file).replace(suffix, '.wav'): file for file in file_list}

# Create mappings
normal_files = get_base_name(wavs_normal)
dysarthric_files = get_base_name(wavs_dys, suffix='_aug.wav')

# Find common files and pair them
train_files = []
for base_name, dys_file in dysarthric_files.items():
    normal_file = normal_files.get(base_name)
    if normal_file:
        train_files.append((dys_file, normal_file))

# Split data into train and validation sets
limit = int(RATE_DB * len(train_files))
train_files_limit = train_files[:limit]
part_train = int(0.8 * len(train_files_limit))
train_files_part = train_files_limit[:part_train]
val_files_part = train_files_limit[part_train:]

        
# Parâmetros
no_of_channels = 3  # Espectrograma color
SIZE = 128  # Tamanho do espectrograma
INPUT_DIM = (SIZE, SIZE, no_of_channels)  # Espectrograma como imagem 2D
LATENT_DIM = 64  # Dimensão do vetor latente

# Função de amostragem
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Pré-processamento de áudio
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SIZE)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    return spectrogram

# Criação de pseudo-rótulos
def create_pseudo_labels(spectrograms, n_clusters=50):
    flattened_spectrograms = [spec.flatten() for spec in spectrograms]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_spectrograms)
    return kmeans.labels_

# Camada de Transformer

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, value_dim=None, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = Dropout(dropout)
        
        self.query_dense = Dense(num_heads * key_dim)
        self.key_dense = Dense(num_heads * key_dim)
        self.value_dense = Dense(num_heads * self.value_dim)
        self.output_dense = Dense(key_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, training):
        batch_size = tf.shape(query)[0]
        height = tf.shape(query)[1]
        width = tf.shape(query)[2]

        # Flatten spatial dimensions
        query = tf.reshape(query, (batch_size, height * width, self.key_dim))
        key = tf.reshape(key, (batch_size, height * width, self.key_dim))
        value = tf.reshape(value, (batch_size, height * width, self.value_dim))

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, height * width, self.num_heads * self.value_dim))

        # Reshape to original spatial dimensions
        attention_output = tf.reshape(attention_output, (batch_size, height, width, self.num_heads * self.value_dim))

        output = self.output_dense(attention_output)
        return output
        
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Apply multi-head attention
        attn_output = self.att(inputs, inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        
        # Layer normalization and residual connection
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Layer normalization and residual connection
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

# Encoder
encoder_input = tf.keras.Input(shape=INPUT_DIM, dtype=tf.float32)
num_conv = {0: {'filters': 256, 'kernel_size': 3, 'strides': 2},
            1: {'filters': 128, 'kernel_size': 3, 'strides': 2},
            2: {'filters': 64, 'kernel_size': 3, 'strides': 2},
            3: {'filters': 32, 'kernel_size': 3, 'strides': 1}
}

encoder = encoder_input
for layer_num, layer_data in num_conv.items():
    encoder = layers.Conv2D(layer_data['filters'], layer_data['kernel_size'], layer_data['strides'], padding='same')(encoder)
    encoder = layers.LeakyReLU(alpha=0.2)(encoder)
    encoder = layers.BatchNormalization(axis=-1)(encoder)

# Adicionar camadas de Transformer
embed_dim = 32  # Dimensão do embedding
num_heads = 2  # Número de cabeças de atenção
ff_dim = 32  # Dimensão da feed-forward network
encoder = TransformerBlock(embed_dim, num_heads, ff_dim)(encoder)

flatten = layers.Flatten()(encoder)
z_mean = layers.Dense(LATENT_DIM)(flatten)
z_log_var = layers.Dense(LATENT_DIM)(flatten)
z = layers.Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_var])

encoder_model = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

# Decoder
decode_input = tf.keras.Input(shape=(LATENT_DIM,))
decoder = layers.Dense(np.prod(encoder.shape[1:]))(decode_input)
decoder = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(decoder)

for layer_num, layer_data in reversed(sorted(num_conv.items())):
    decoder = layers.Conv2DTranspose(layer_data['filters'], layer_data['kernel_size'], layer_data['strides'], padding='same')(decoder)
    decoder = layers.LeakyReLU(alpha=0.2)(decoder)
    decoder = layers.BatchNormalization(axis=-1)(decoder)

decoder = layers.Conv2DTranspose(no_of_channels, 3, padding='same')(decoder)
decoder_output = layers.Activation('sigmoid')(decoder)
decoder_model = Model(decode_input, decoder_output, name='decoder')

# VAE model
vae_input = encoder_input
z_mean, z_log_var, z = encoder_model(vae_input)
vae_output = decoder_model(z)
vae_model = Model(vae_input, vae_output, name='vae')

# Função de perda
reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(vae_input, vae_output))
reconstruction_loss *= SIZE * SIZE * no_of_channels
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae_model.add_loss(vae_loss)
vae_model.compile(optimizer='adam')

# Function to visualize a sample of data
def visualize_data(generator):
    # Get a batch of data
    batch = next(generator)
    x, y = batch

    # Ensure we have the correct dimensions
    if x.ndim == 4 and y.ndim == 4:
        # Visualize the first image of the batch
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(x[0, :, :, 0], cmap='viridis')
        ax[0].set_title('Input Data (First Channel)')
        ax[1].imshow(y[0, :, :, 0], cmap='viridis')
        ax[1].set_title('Output Data (First Channel)')
        plt.show()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(x[:, :, 0], cmap='viridis')
        ax[0].set_title('Input Data')
        ax[1].imshow(y[:, :, 0], cmap='viridis')
        ax[1].set_title('Output Data')
        plt.show()
    plt.close(fig)

# Define generators
def train_generator():
    for disartric_file, normal_file in train_files_part:
        x = load_wav(disartric_file)
        y = load_wav(normal_file)
        yield x, y

def val_generator():
    for disartric_file, normal_file in val_files_part:
        x = load_wav(disartric_file)
        y = load_wav(normal_file)
        yield x, y

train_dataset = tf.data.Dataset.from_generator(train_generator, output_signature=(
    tf.TensorSpec(shape=(SIZE, SIZE, no_of_channels), dtype=tf.float32),
    tf.TensorSpec(shape=(SIZE, SIZE, no_of_channels), dtype=tf.float32)
)).batch(BATCH)

val_dataset = tf.data.Dataset.from_generator(val_generator, output_signature=(
    tf.TensorSpec(shape=(SIZE, SIZE, no_of_channels), dtype=tf.float32),
    tf.TensorSpec(shape=(SIZE, SIZE, no_of_channels), dtype=tf.float32)
)).batch(BATCH)

# Visualize training data
visualize_data(train_generator())

# Custom callback to visualize a sample spectrogram after each epoch
class VisualizeCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for batch in val_dataset.take(1):
            x, y = batch
            idx = np.random.randint(0, x.shape[0])
            input_spec = x[idx]
            target_spec = y[idx]
            output_spec = self.model.predict(np.expand_dims(input_spec, axis=0))[0]
            plot_spectrograms(input_spec, target_spec, output_spec, epoch + 1, 0)
            
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=5, verbose=1, mode='min'):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode '{mode}' is not supported, use 'min' or 'max'.")

        if mode == 'min':
            self.monitor_op = tf.less
            self.best = float('inf')
        else:
            self.monitor_op = tf.greater
            self.best = -float('inf')

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                if self.verbose > 0:
                    print(f"Restoring model weights from the end of the best epoch: {self.best_epoch + 1}")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")

# ----------------------- Ini Carrega Modelo----------------------

model_path = 'vae_hubert_model.h5'
if os.path.exists(model_path):
    vae_model.load_weights('vae_hubert_model.h5')
    print("Modelo carregado com sucesso!")

# ----------------------- Fim Carrega Modelo----------------------
            
# Use the custom callback in your training process
early_stopping_cb = CustomEarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Callbacks for saving the best model and adjusting learning rate
#checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("vae_best_model", save_best_only=True, monitor='val_loss')
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('vae_hubert_model.h5', save_best_only=True, save_weights_only=False)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training the model
vae_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_cb, reduce_lr_cb, VisualizeCallback()])


gc.collect()

'''
def test_autoencoder(autoencoder, test_files, spectrogram_dir):
    for file in tqdm(test_files, desc="Processing files"):
        x = load_wav(file)
        x = np.expand_dims(x, axis=0)
        reconstructed = autoencoder(x, training=False)
        reconstructed = np.squeeze(reconstructed, axis=0)

        spectrogram_path = os.path.join(spectrogram_dir, os.path.basename(file).replace('.wav', '.png'))
        save_spectrogram(reconstructed, spectrogram_path)
'''

def test_autoencoder(autoencoder, test_files, spectrogram_dir):
    for file in tqdm(test_files, desc="Processing files"):
        x = load_wav(file)
        x = np.expand_dims(x, axis=0)
        reconstructed = autoencoder(x, training=False)
        reconstructed = np.squeeze(reconstructed, axis=0)

        spectrogram_path = os.path.join(spectrogram_dir, os.path.basename(file).replace('.wav', '.png'))
        save_spectrogram(x[0], reconstructed, spectrogram_path)


# Test the autoencoder with test data
test_autoencoder(vae_model, wavs_test, spectrogram_dir)
