import os
import time
import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import shutil
import math

# TODO: implement other architectures (smaller AutoEncoder, Transformer, SelectionGAN) and think about which hyperparameters to also check (dropout, skip)
# ideas: compare to untrained evaluation


##### SELECT YOUR FIXED SETTINGS (SAME FOR EACH CONFIGURATION) HERE


#DATASET_PATH = '/data/dataset_no_unwrapping_synthetic_vortex_20000img_phasemaskversion2'
DATASET_PATH = r'C:\Users\burckhardsv\Lokale_Dateien\Dataset\Test_Dataset_Vortex_extra_small'
N_EPOCHS = 1  # I suggest 50
SHOW_RESULTS_EACH_N_EPOCHS = 999  # I suggest 3 (but a very high value for ommitting intermediate results (improves efficiency))
EVALUATE_EACH_N_EPOCHS = 1  # I suggest 1, set for example to 2 to increase efficiency by evaluating less often
N_EPOCHS_WITHOUT_IMPROVEMENT_FOR_EARLY_STOPPING = 3  # after how much epochs without improvements in validation loss to early stop (set to float('inf') for no early stopping)
PATH_TO_STORE_RESULTS = 'results_hyperparameter_optimization'  # set to '' for inline-plots
PRETRAINED_WEIGHTS = ''  # path to model weights, e.g. 'results/2024-01-17_09-26-22-040485/checkpoints/MODEL/epoch_8', set to '' to train model from scratch. Note that 'MODEL' will be replaced by 'generator' and discriminator' automatically and needs to be kept as a placeholder
USE_GENERATOR_ONLY = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0" for using GPU:0, "" for "GPU:1", "0,1" for both

SKIP_FIRST_N_CONFIGS = 0  # default is 0, increase if you want to continue an earlier tuning run

### SELECT YOUR HYPERPARAMETERS HERE

configurations = []

for batch_size in [64, 1]:
    for learning_rate, beta_1 in [(2e-4, 0.5), (1e-3, 0.9), (1e-5, 0.9)]:
        for use_bias_term in [True, False]:
            for use_l2_loss in [False, True]:
                for lambda_ in [1]:
                    for use_custom_distance_measure in [False]:
                        for use_separate_conv_for_nf_and_ff in [False, True]:
                            for use_skip_connections in [True, False]:
                                batches_to_plot = round(128 / batch_size)
                                configurations.append({
                                    'BATCH_SIZE': batch_size,
                                    'LAMBDA': lambda_,
                                    'USE_L2_LOSS': use_l2_loss,
                                    'N_EPOCHS': N_EPOCHS,
                                    'BATCHES_TO_PLOT': batches_to_plot,
                                    'USE_BIAS_TERM': use_bias_term,
                                    'USE_CUSTOM_DISTANCE_MEASURE': use_custom_distance_measure,
                                    'USE_SEPARATE_CONV_FOR_NF_AND_FF': use_separate_conv_for_nf_and_ff,
                                    'LEARNING_RATE': learning_rate,
                                    'BETA_1': beta_1,
                                    'USE_SKIP_CONNECTIONS': use_skip_connections
                                })


### PREPARE LOGGING

script_started_at = time.time()
TIMESTAMP = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '_')
path_to_store_results = f'{PATH_TO_STORE_RESULTS}/{TIMESTAMP}'
os.mkdir(path_to_store_results)

def log_to_main_logfile(message):
    with open(f'{PATH_TO_STORE_RESULTS}/{TIMESTAMP}/log.txt', 'a') as f:
        f.write(message + '\n')

with open(f'{path_to_store_results}/configurations.txt', 'w') as f:
    f.write(f'dataset = {DATASET_PATH}\n\n' + str([config for config in enumerate(configurations)]))

with open(f'{path_to_store_results}/results_summary.csv', 'w') as f:
    f.write('folder,best_epoch,train_loss_g_total,val_loss_g_total,test_loss_g_total,train_ssim,val_ssim,test_ssim\n')


### LOOP THROUGH ALL HYPERPARAMETER CONFIGURATIONS

for config_idx, configuration in enumerate(configurations):

    if config_idx < SKIP_FIRST_N_CONFIGS:
        continue

    ### PREPARE LOGGING FOR THIS SPECIFIC CONFIGURATION

    timestamp = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '_')
    os.mkdir(f'{path_to_store_results}/{timestamp}')
    os.mkdir(f'{path_to_store_results}/{timestamp}/example_images')
    os.mkdir(f'{path_to_store_results}/{timestamp}/history')
    def log(message, end='\n'):
        with open(f'{path_to_store_results}/{timestamp}/log.txt', 'a') as f:
            f.write(message + end)
    

    ### DEFINE DISTANCE BETWEEN PIXELS
    def pixelwise_distance(img1, img2):
        if configuration['USE_CUSTOM_DISTANCE_MEASURE']:
            abs_dist = tf.abs(img2 - img1)
            return tf.minimum(abs_dist, tf.abs(abs_dist - math.pi))
        return tf.abs(img1 - img2)


    ### LOAD AND PREPROCESS DATASET

    def preprocess_image(img):
        return img / 255.0

    def folder_to_dataloader(folder):
        return tf.keras.preprocessing.image_dataset_from_directory(
            directory=folder,
            color_mode='grayscale',
            labels=None,
            shuffle=False,
            image_size=[256, 256],
            batch_size=configuration['BATCH_SIZE']
        ).map(preprocess_image)

    dataset = {}  # structure: dataset['train'], dataset['test'], , dataset['val'] each tuples (nearfield_img, farfield_img, phasemask_img)

    for subset in ['train', 'test', 'val']:
        nearfield = folder_to_dataloader(f'{DATASET_PATH}/{subset}/beam_nf')
        farfield = folder_to_dataloader(f'{DATASET_PATH}/{subset}/beam_ff')
        phasemask = folder_to_dataloader(f'{DATASET_PATH}/{subset}/phasemask')
        dataset[subset] = tf.data.Dataset.zip((nearfield, farfield, phasemask))
    

    ### DEFINE MODEL

    def downsample(n_filters, kernel_size, apply_batchnorm=True):
        
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(
                n_filters,
                kernel_size,
                strides=2,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(0., 0.02),
                use_bias=configuration['USE_BIAS_TERM']
            )
        )

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(n_filters, kernel_size, apply_dropout=False):

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(
                n_filters,
                kernel_size,
                strides=2,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(0., 0.02),
                use_bias=configuration['USE_BIAS_TERM']
            )
        )

        result.add(tf.keras.layers.BatchNormalization())  # TODO: why do we do this always when upsampling but only sometimes when downsampling?

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))  # TODO: maybe rethink about dropouts

        result.add(tf.keras.layers.ReLU())

        return result


    def GeneratorConcatenteNfAndFfRepresentation():
        inputs = tf.keras.layers.Input(shape=[256, 256, 2])
        inputs_per_channel = tf.split(inputs, num_or_size_splits=2, axis=-1)

        down_stack = [[
            downsample(32, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 32)
            downsample(64, 4),  # (batch_size, 64, 64, 64)
            downsample(128, 4),  # (batch_size, 32, 32, 128)
            downsample(256, 4),  # (batch_size, 16, 16, 256)
            downsample(256, 4),  # (batch_size, 8, 8, 256)
            downsample(256, 4),  # (batch_size, 4, 4, 256)
            downsample(256, 4),  # (batch_size, 2, 2, 256)
            downsample(256, 4),  # (batch_size, 1, 1, 256)
        ] for _ in range(2)]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 256, 256, 1)

        if configuration['USE_SKIP_CONNECTIONS']:
            skips = []
        x = None
        for channel_idx in range(2):
            x_channel = inputs_per_channel[channel_idx]
            for downsampling_step, down in enumerate(down_stack[channel_idx]):
                x_channel = down(x_channel)
                if configuration['USE_SKIP_CONNECTIONS']:
                    if channel_idx == 0:
                        skips.append(x_channel)
                    else:
                        skips[downsampling_step] = tf.keras.layers.Concatenate()([skips[downsampling_step], x_channel])
            if channel_idx == 0:
                x = x_channel
            else:
                x = tf.keras.layers.Concatenate()([x, x_channel])

        if configuration['USE_SKIP_CONNECTIONS']:
            skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        if configuration['USE_SKIP_CONNECTIONS']:
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])
        else:
            for up in up_stack:
                x = up(x)

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def GeneratorChannelMerge():
        inputs = tf.keras.layers.Input(shape=[256, 256, 2])

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 256, 256, 1)

        x = inputs / 2

        # Downsampling through the model
        if configuration['USE_SKIP_CONNECTIONS']:
            skips = []
        for down in down_stack:
            x = down(x)
            if configuration['USE_SKIP_CONNECTIONS']:
                skips.append(x)

        if configuration['USE_SKIP_CONNECTIONS']:
            skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        if configuration['USE_SKIP_CONNECTIONS']:
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = tf.keras.layers.Concatenate()([x, skip])
        else:
            for up in up_stack:
                x = up(x)

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def Generator(use_separate_conv_for_nf_and_ff):
        if use_separate_conv_for_nf_and_ff:
            return GeneratorConcatenteNfAndFfRepresentation()
        return GeneratorChannelMerge()


    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 2], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, 2)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    

    ### SPECIFY TRAINING AND EVALUATION PROCEDURE

    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def generator_loss(gen_output, target, disc_generated_output=None):
        if disc_generated_output is None:
            gan_loss = 0
        else:
            gan_loss = bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)
        if configuration['USE_L2_LOSS']:
            l_loss = tf.reduce_mean(tf.square(pixelwise_distance(target, gen_output)))  # MSE (L2 loss)
        else:
            l_loss = tf.reduce_mean(pixelwise_distance(target, gen_output))  # MAE (L1 loss)
        total_generator_loss = gan_loss + (configuration['LAMBDA'] * l_loss)
        return total_generator_loss, gan_loss, l_loss


    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = bce_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = bce_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_discriminator_loss = real_loss + generated_loss
        return total_discriminator_loss

    generator = Generator(configuration['USE_SEPARATE_CONV_FOR_NF_AND_FF'])
    if not USE_GENERATOR_ONLY:
        discriminator = Discriminator()

    if PRETRAINED_WEIGHTS != '':
        generator.load_weights(PRETRAINED_WEIGHTS.replace('MODEL', 'generator'))
        if not USE_GENERATOR_ONLY:
            discriminator.load_weights(PRETRAINED_WEIGHTS.replace('MODEL', 'discriminator'))

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=configuration['LEARNING_RATE'], beta_1=configuration['BETA_1'])
    if not USE_GENERATOR_ONLY:
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=configuration['LEARNING_RATE'], beta_1=configuration['BETA_1'])


    def generate_images(epoch, generator, val_dataset_to_plot):

        img_idx = 0

        for batch in val_dataset_to_plot:
            
            nearfields, farfields, phasemasks = batch

            input_images = tf.concat([nearfields, farfields], axis=3)
            generated_images = generator(input_images, training=True)  # TODO: adapt?

            title = ['Input Nearfield', 'Input Farfield', 'Predicted Phasemask', 'Target Phasemask']

            for images in zip(nearfields, farfields, generated_images, phasemasks):

                img_idx += 1
                images = list(images)

                plt.figure(figsize=(15, 15))
                for i in range(len(title)):
                    plt.subplot(1, len(title), i+1)
                    plt.title(title[i])
                    plt.imshow(images[i], cmap='gray')
                    plt.axis('off')
                
                filename_prefix = 'epoch_' if epoch != 'final-performance' else ''
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/{filename_prefix}{epoch}__img_{img_idx}.png')
                plt.close()


    def evaluate(generator, val_dataset, discriminator=None):

        gen_total_loss_sum = 0
        gen_gan_loss_sum = 0
        gen_l_loss_sum = 0
        disc_loss_sum = 0
        ssim_sum = 0

        for batch in val_dataset:
            
            nearfields, farfields, phasemasks = batch
            input_images = tf.concat([nearfields, farfields], axis=3)

            generated_images = generator(input_images, training=True)  # TODO: adapt?

            if discriminator is not None:
                disc_real_output = discriminator([input_images, phasemasks], training=True)
                disc_generated_output = discriminator([input_images, generated_images], training=True)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                gen_total_loss, gen_gan_loss, gen_l_loss = generator_loss(generated_images, phasemasks, disc_generated_output)
            else:
                disc_loss = 0
                gen_total_loss, gen_gan_loss, gen_l_loss = generator_loss(generated_images, phasemasks)

            ssim_score = float(tf.reduce_mean(tf.image.ssim(phasemasks, generated_images, max_val=1.0)))

            gen_total_loss_sum += gen_total_loss
            gen_gan_loss_sum += gen_gan_loss
            gen_l_loss_sum += gen_l_loss
            disc_loss_sum += disc_loss
            ssim_sum += ssim_score

        gen_total_loss = float(gen_total_loss_sum) / len(val_dataset)
        gen_gan_loss = float(gen_gan_loss_sum) / len(val_dataset)
        gen_l_loss = float(gen_l_loss_sum) / len(val_dataset)
        disc_loss = float(disc_loss_sum) / len(val_dataset)
        ssim_score = float(ssim_sum) / len(val_dataset)

        return gen_total_loss, gen_gan_loss, gen_l_loss, disc_loss, ssim_score
    

    def save_weights(epoch, generator, discriminator=None, delete_previous=False):
        if delete_previous:
            folder = f'{path_to_store_results}/{timestamp}/checkpoints'
            try:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
            except:
                pass  # no checkpoints yet
        
        generator.save_weights(f'{path_to_store_results}/{timestamp}/checkpoints/generator/epoch_{epoch}')
        if discriminator is not None:
            discriminator.save_weights(f'{path_to_store_results}/{timestamp}/checkpoints/discriminator/epoch_{epoch}')
                    

    ### DEFINE TRAINING ROUTINE

    @tf.function
    def train_step(input_images, targets):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            gen_output = generator(input_images, training=True)

            if not USE_GENERATOR_ONLY:
                disc_real_output = discriminator([input_images, targets], training=True)
                disc_generated_output = discriminator([input_images, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l_loss = generator_loss(gen_output, targets, (None if USE_GENERATOR_ONLY else disc_generated_output))
            if not USE_GENERATOR_ONLY:
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
            if not USE_GENERATOR_ONLY:
                discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            if not USE_GENERATOR_ONLY:
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            ssim_score = float(tf.reduce_mean(tf.image.ssim(targets, gen_output, max_val=1.0)))

            return gen_total_loss, gen_gan_loss, gen_l_loss, (0 if USE_GENERATOR_ONLY else disc_loss), ssim_score


    def fit(train_dataset, val_dataset, n_epochs):

        history = {
            'epochs': {'train': [], 'val': []},
            'generator_total_loss': {'train': [], 'val': []},
            'generator_gan_loss': {'train': [], 'val': []},
            'generator_L_loss': {'train': [], 'val': []},
            'discriminator_loss': {'train': [], 'val': []},
            'ssim': {'train': [], 'val': []}
        }

        val_dataset_to_plot = val_dataset.take(configuration['BATCHES_TO_PLOT'])
        consecutive_epochs_without_improvements = 0
        best_val_loss = float('inf')
        training_started_at = time.time()

        for epoch in range(1, n_epochs + 1):

            epoch_started_at = time.time()

            gen_total_loss_sum = 0
            gen_gan_loss_sum = 0
            gen_l_loss_sum = 0
            disc_loss_sum = 0
            ssim_sum = 0
            
            for batch in train_dataset:
                
                nearfields, farfields, phasemasks = batch
                input_images = tf.concat([nearfields, farfields], axis=3)
                target_images = phasemasks

                gen_total_loss, gen_gan_loss, gen_l_loss, disc_loss, ssim_score = train_step(input_images, target_images)

                gen_total_loss_sum += gen_total_loss
                gen_gan_loss_sum += gen_gan_loss
                gen_l_loss_sum += gen_l_loss
                disc_loss_sum += disc_loss
                ssim_sum += ssim_score

            history['epochs']['train'].append(epoch)
            history['generator_total_loss']['train'].append(float(gen_total_loss_sum) / len(train_dataset))
            history['generator_gan_loss']['train'].append(float(gen_gan_loss_sum) / len(train_dataset))
            history['generator_L_loss']['train'].append(float(gen_l_loss_sum) / len(train_dataset))
            history['discriminator_loss']['train'].append(float(disc_loss_sum) / len(train_dataset))
            history['ssim']['train'].append(float(ssim_sum) / len(train_dataset))
            
            # log progress
            delta_t = time.time() - epoch_started_at
            seconds = round(delta_t % 60)
            epoch_took = str(round(delta_t // 60)) + ':' + ('0' if seconds < 10 else '') + str(seconds)
            delta_t = time.time() - training_started_at
            seconds = round(delta_t % 60)
            config_took = str(round(delta_t // 60)) + ':' + ('0' if seconds < 10 else '') + str(seconds)
            delta_t = time.time() - script_started_at
            seconds = round(delta_t % 60)
            total_took = str(round(delta_t // 60)) + ':' + ('0' if seconds < 10 else '') + str(seconds)
            log(f'Epoch {epoch} / {n_epochs} completed in {epoch_took} min, total {config_took} min')
            log_to_main_logfile(f'Epoch {epoch}/{n_epochs} of config {config_idx+1}/{len(configurations)} completed in {epoch_took} min, config running for {config_took} min, script for {total_took} min')

            if epoch % SHOW_RESULTS_EACH_N_EPOCHS == 0:
                generate_images(epoch, generator, val_dataset_to_plot)

            if epoch % EVALUATE_EACH_N_EPOCHS == 0:
                
                gen_total_loss, gen_gan_loss, gen_l_loss, disc_loss, ssim_score = evaluate(generator, val_dataset, (None if USE_GENERATOR_ONLY else discriminator))
                
                history['epochs']['val'].append(epoch)
                history['generator_total_loss']['val'].append(gen_total_loss)
                history['generator_gan_loss']['val'].append(gen_gan_loss)
                history['generator_L_loss']['val'].append(gen_l_loss)
                history['discriminator_loss']['val'].append(disc_loss)
                history['ssim']['val'].append(ssim_score)
                log('\nCurrent performance:')
                
                for metric in history:
                    if metric == 'epochs':
                        continue
                    for subset in ['train', 'val']:
                        log(metric + '_' + subset + ' = ' + str(history[metric][subset][-1]))
                log('')

                # check for early stopping and save weights
                if history['generator_total_loss']['val'][-1] < best_val_loss:
                    best_val_loss = history['generator_total_loss']['val'][-1]
                    consecutive_epochs_without_improvements = 0
                    save_weights(epoch, generator, (None if USE_GENERATOR_ONLY else discriminator), delete_previous=True)
                else:
                    consecutive_epochs_without_improvements += EVALUATE_EACH_N_EPOCHS
                    if consecutive_epochs_without_improvements >= N_EPOCHS_WITHOUT_IMPROVEMENT_FOR_EARLY_STOPPING:
                        break

        save_weights(epoch, generator, (None if USE_GENERATOR_ONLY else discriminator), delete_previous=False)
            
        return history


    ### TRAIN MODEL

    history = fit(dataset['train'], dataset['val'], n_epochs=N_EPOCHS)
    log('history = ' + str(history) + '\n')


    ### EVALUATE MODEL

    # reset model to best-performing state
    best_idx_val = np.argmin(history['generator_total_loss']['val'])
    best_epoch = history['epochs']['val'][best_idx_val]
    path_to_best_weights = f'{path_to_store_results}/{timestamp}/checkpoints/MODEL/epoch_{best_epoch}'
    generator.load_weights(path_to_best_weights.replace('MODEL', 'generator'))
    if not USE_GENERATOR_ONLY:
        discriminator.load_weights(path_to_best_weights.replace('MODEL', 'discriminator'))

    # analyze performance
    final_test_performance = {}
    for subset in ['train', 'val', 'test']:
        gen_total_loss, gen_gan_loss, gen_l_loss, disc_loss, ssim_score = evaluate(generator, dataset[subset], (None if USE_GENERATOR_ONLY else discriminator))
        for metric in ['gen_total_loss', 'gen_gan_loss', 'gen_l_loss', 'disc_loss', 'ssim_score']:
            log(f'final {subset} {metric} = {eval(metric)}')
            final_test_performance[metric] = eval(metric)
        generate_images('final-performance', generator, dataset[subset].take(configuration['BATCHES_TO_PLOT']))


    ### GENERATE PLOTS SHOWING TRAINING HISTORY

    # create one plot
    for metric in history:
        if metric == 'epochs':
            continue
        plt.plot(history['epochs']['train'], history[metric]['train'], label = metric + '_TRAIN')
        plt.plot(history['epochs']['val'], history[metric]['val'], label = metric + '_VAL')
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.xlabel('epoch')

    plt.savefig(f'{path_to_store_results}/{timestamp}/history/results_total.png', dpi=300)
    plt.close()

    # create single plots
    for metric in history:
        if metric == 'epochs':
            continue
        plt.plot(history['epochs']['train'], history[metric]['train'], label='train')
        plt.plot(history['epochs']['val'], history[metric]['val'], label='val')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')

        plt.savefig(f'{path_to_store_results}/{timestamp}/history/results_{metric}.png', dpi=300)
        plt.close()
    
    
    ### LOGGING
    
    best_idx_train = best_epoch - 1
    loss_train = history['generator_total_loss']['train'][best_idx_train]
    loss_val = history['generator_total_loss']['val'][best_idx_val]
    ssim_train = history['ssim']['train'][best_idx_train]
    ssim_val = history['ssim']['val'][best_idx_val]
    with open(f'{path_to_store_results}/results_summary.csv', 'a') as f:
        f.write(f"{timestamp},{best_epoch},{loss_train},{loss_val},{final_test_performance['gen_total_loss']},{ssim_train},{ssim_val},{final_test_performance['ssim_score']}\n")
