import os
import time
import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import shutil
import math
import sys
import cv2
from matplotlib import pyplot as plt
import yaml

from experimental_sven.sonam.GAN_hyperparameter_tuning_20000img_Sonam_Code import PATH_TO_STORE_RESULTS

# TODO: add noise to beam images

#DATASET_PATH = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Small_Other_Datasets\Dataset_Ditzingen_Gaussain_Mini_512'
#DATASET_PATH = r'C:\Users\burckhardsv\Lokale_Dateien\Dataset\Test_Dataset_Vortex_extra_small'
DATASET_PATH = r'/kaggle/input/jikai-wang-test/unwrapped_simulated_test'

###################################################### Do not touch #############################################################################
# PATH_TO_STORE_RESULTS = 'results_hyperparameter_optimization'
PATH_TO_STORE_RESULTS =  '/kaggle/working/experimental_sven/results_hyperparameter_optimization'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0" for using GPU:0, "1" for GPU:1, "0,1" for both

########################################################### Configuration ########################################################
def load_hyperparameters(filepath):
    with open(filepath, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

filepath = '/kaggle/working/experimental_sven/Autoencoder/config.yaml'
hyperparams = load_hyperparameters(filepath)


# Fix Parameters
EVALUATE_EACH_N_EPOCHS = hyperparams['EVALUATE_EACH_N_EPOCHS']
N_EPOCHS_WITHOUT_IMPROVEMENT_FOR_EARLY_STOPPING = hyperparams['N_EPOCHS_WITHOUT_IMPROVEMENT_FOR_EARLY_STOPPING']
PRETRAINED_WEIGHTS = hyperparams['PRETRAINED_WEIGHTS']
SHOW_RESULTS_EACH_N_EPOCHS = hyperparams['SHOW_RESULTS_EACH_N_EPOCHS']
SKIP_FIRST_N_CONFIGS = hyperparams['SKIP_FIRST_N_CONFIGS']
ORIGINAL_PHASEMASK_SIZE = hyperparams['ORIGINAL_PHASEMASK_SIZE']


IMAGE_SIZE = hyperparams['IMAGE_SIZE']
MODUS = hyperparams['MODUS']
MODEL_NUMBER = hyperparams['MODEL_NUMBER']
# Hyperparameters
N_EPOCHS = hyperparams['N_EPOCHS']

BEST_CONFIGURATION_ONLY = hyperparams['BEST_CONFIGURATION_ONLY']
BEST_CONFIGURATION = hyperparams['BEST_CONFIGURATION']
EVALUATION_METHOD = hyperparams['EVALUATION_METHOD']
STRIDES = hyperparams['STRIDES']
PADDING = hyperparams['PADDING']

i=1

##########################################################################################################
configurations = []

for batch_size in [1]:
    for learning_rate, beta_1 in [(2e-4, 0.5),(1e-4, 0.5),(2e-4, 0.9)]: # [(2e-4, 0.5), (1e-5, 0.5),(2e-4, 0.9),(1e-5, 0.9)]:
        for use_bias_term in [True]:
            for use_l2_loss in [True]: # only MSE before [False,True]
                for DROPOUT_VALUE in [0.2,0.5,0.6]: # before only 0.5
                        for KERNEL_SIZE in[3,4]:
                            for WEIGHT_DECAY in [0, 0.004]: # default tensorflow adamW
                                for use_skip_connections in [True]:
                                    batches_to_plot = round(128 / batch_size)
                                    configurations.append({
                                        'BATCH_SIZE': batch_size,
                                        'USE_L2_LOSS': use_l2_loss,
                                        'N_EPOCHS': N_EPOCHS,
                                        'BATCHES_TO_PLOT': batches_to_plot,
                                        'USE_BIAS_TERM': use_bias_term,
                                        'LEARNING_RATE': learning_rate,
                                        'BETA_1': beta_1,
                                        'USE_SKIP_CONNECTIONS': use_skip_connections,
                                        'DROPOUT_VALUE': DROPOUT_VALUE,
                                        'KERNEL_SIZE':KERNEL_SIZE,
                                        'WEIGHT_DECAY': WEIGHT_DECAY,
                                    })

if BEST_CONFIGURATION_ONLY == True:
    configurations= [configurations[BEST_CONFIGURATION]]
else:  
    print(f"Use all Configurations")

########################################################## Prepare Logging ###################################################################################
if MODUS==0:
    script_started_at = time.time()
    TIMESTAMP = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '_')
    path_to_store_results = f'{PATH_TO_STORE_RESULTS}/{TIMESTAMP}'
    os.makedirs(path_to_store_results)

    with open(f'{path_to_store_results}/configurations.txt', 'w') as f:
        f.write(f'dataset = {DATASET_PATH}\n\n' + str([config for config in enumerate(configurations)]))
        f.write(f'Strides: {STRIDES} | Ssim evaluation metric: {EVALUATION_METHOD} | Epochs: {N_EPOCHS} | Model_number: {MODEL_NUMBER}' )

    with open(f'{path_to_store_results}/results_summary.csv', 'w') as f:
        f.write('folder,best_epoch,l1_loss_train,l1_loss_val,l1_loss_test,l1_loss_test_without_dropout,l2_loss_train,l2_loss_val,l2_loss_test,l2_loss_test_without_dropout,ssim_train,ssim_val,ssim_test,ssim_test_without_dropout\n') 
####################################################### Functions #####################################################################################
def get_timestamp_as_string():
    return str(datetime.datetime.now()).split('.')[0]

def log_to_main_logfile(message):
    with open(f'{PATH_TO_STORE_RESULTS}/{TIMESTAMP}/log.txt', 'a') as f:
        f.write(f'[{get_timestamp_as_string()}] ' + message + '\n')

def get_mask(image):
    #print(f"Image Shape  {tf.shape(image)}")
    """Method to get a mask to only use this area later in the ssim calculation"""
    dims = image.shape[:]
    #print(dims)
    mask = np.zeros(dims, dtype=np.float32)
    center = (dims[2]//2, dims[1]//2)
    radius = min(center) 
    Y, X = np.ogrid[:dims[1], :dims[2]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    #print(radius, dist_from_center.shape)
    for i in range(len(mask)):
        mask[i][dist_from_center <= radius] = 1.0
    if EVALUATION_METHOD == 0:
        #Safe mask for debug:
        mask_path = f'{path_to_store_results}/{timestamp}/example_images/mask_.png'
        if not os.path.exists(mask_path):
            plt.figure(figsize=(6.67, 6.67), dpi=300)
            plt.switch_backend('agg')
            plt.imshow(mask[0, :, :, 0], cmap='gray', interpolation='none')
            plt.axis('off')
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close('all')
    return mask

def ssim_specific(img1, img2, mask, max_val=1.0):  
    """Only use the important area for our usecase to calculate ssim"""
    #mask = tf.expand_dims(mask, axis=-1)
    min_value = tf.reduce_min(img1)
    img1_masked, img2_masked = [tf.where(mask > 0, x, min_value) for x in [img1, img2]]
    img1_masked, img2_masked = [tf.expand_dims(x, axis=-1) if x.shape.ndims == 2 else x for x in [img1_masked, img2_masked]]
    ssim_index = tf.image.ssim(img1_masked, img2_masked, max_val=max_val)

    return ssim_index 

def pixelwise_distance(img1, img2, configuration):
    return tf.abs(img1 - img2)

def preprocess_image(img):
        return img / 255.0

def resize(size,img):
    current_size = img.shape
    #print(current_size)
    if current_size[0]< size[0]:
        # upscale the image
        img_original_size = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_CUBIC) 
        return img_original_size
    elif current_size[0] == size[0]:
        print("Already in the original size")
        #img_original_size = img
        # save_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask_predicted_test.png'
        # cv2.imwrite(save_path, img)
        return img
    else:
        raise ValueError("Something wrong with the resize downscaling after training.")


def folder_to_dataloader(folder, configuration):
    """structure: dataset['train'], dataset['test'], , dataset['val'] each tuples (nearfield_img, farfield_img, phasemask_img)"""

    return tf.keras.preprocessing.image_dataset_from_directory(
        directory=folder,
        color_mode='grayscale',
        labels=None,
        shuffle=False,
        image_size=[IMAGE_SIZE[0], IMAGE_SIZE[1]],  
        batch_size=configuration['BATCH_SIZE']
    ).map(preprocess_image)

def log(message, end='\n'):
            with open(f'{path_to_store_results}/{timestamp}/log.txt', 'a') as f:
                f.write(f'[{get_timestamp_as_string()}] ' + message + end)
##################################################################### DEFINE MODEL ##############################################################################
def downsample(n_filters, kernel_size, configuration, apply_batchnorm=True):
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            strides=STRIDES, 
            padding=PADDING, 
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=configuration['USE_BIAS_TERM']
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(n_filters, kernel_size, configuration, apply_dropout=False):

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            n_filters,
            kernel_size,
            strides=STRIDES,
            padding=PADDING,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=configuration['USE_BIAS_TERM']
        )
    )

    result.add(tf.keras.layers.BatchNormalization())  # TODO: why do we do this always when upsampling but only sometimes when downsampling?

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(DROPOUT_VALUE))  # TODO: maybe rethink about dropouts

    result.add(tf.keras.layers.ReLU())

    return result

def autoencoder_split_256x256(configuration):
    inputs = tf.keras.layers.Input(shape=[256, 256, 2])
    inputs_per_channel = tf.split(inputs, num_or_size_splits=2, axis=-1)
    down_stack = [[
        downsample(32, 4,   configuration, apply_batchnorm=False),  # (batch_size, 128, 128, 32)
        downsample(64, 4,   configuration),   # (batch_size, 64, 64, 64)
        downsample(128, 4,  configuration),  # (batch_size, 32, 32, 128)
        downsample(256, 4,  configuration),  # (batch_size, 16, 16, 256)
        downsample(256, 4,  configuration),  # (batch_size, 8, 8, 256)
        downsample(256, 4,  configuration),  # (batch_size, 4, 4, 256)
        downsample(256, 4,  configuration),  # (batch_size, 2, 2, 256)
        downsample(256, 4,  configuration),  # (batch_size, 1, 1, 256)
    ] for _ in range(2)]

    up_stack = [
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4,    configuration),  # (batch_size, 16, 16, 1024)
        upsample(256, 4,    configuration),  # (batch_size, 32, 32, 512)
        upsample(128, 4,    configuration),  # (batch_size, 64, 64, 256)
        upsample(64, 4,     configuration),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, configuration['KERNEL_SIZE'], strides=STRIDES, padding=PADDING, kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 256, 256, 1)

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


def autoencoder_256x256(configuration):
    inputs = tf.keras.layers.Input(shape=[256, 256, 2])
    #print("Test", inputs.shape)
    down_stack = [
        downsample(64, 4,   configuration, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4,  configuration),  # (batch_size, 64, 64, 128)
        downsample(256, 4,  configuration),  # (batch_size, 32, 32, 256)
        downsample(512, 4,  configuration),  # (batch_size, 16, 16, 512)
        downsample(512, 4,  configuration),  # (batch_size, 8, 8, 512)
        downsample(512, 4,  configuration),  # (batch_size, 4, 4, 512)
        downsample(512, 4,  configuration),  # (batch_size, 2, 2, 512)
        downsample(512, 4,  configuration),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 2, 2, 512)
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 4, 4, 512)
        upsample(512, 4,    configuration, apply_dropout=True),  # (batch_size, 8, 8, 512)
        upsample(512, 4,    configuration),  # (batch_size, 16, 16, 512)
        upsample(256, 4,    configuration),  # (batch_size, 32, 32, 256)
        upsample(128, 4,    configuration),  # (batch_size, 64, 64, 128)
        upsample(64, 4,     configuration),  # (batch_size, 128, 128, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, configuration['KERNEL_SIZE'], strides=STRIDES, padding=PADDING, kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 256, 256, 1)

    x = inputs / 2
    
    # Downsampling through the model
    if configuration['USE_SKIP_CONNECTIONS']:
        skips = []
    for down in down_stack:
        x = down(x)
        # print("Downsample")
        # print(x)
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
            # print("Upsample")
            # print(x)

    x = last(x)
    # print('Kernel_size', configuration['KERNEL_SIZE'])
    # print("LAst:" , x.shape)
    # print(inputs.shape)
    return tf.keras.Model(inputs=inputs, outputs=x)

def autoencoder_512x512(configuration):
    inputs = tf.keras.layers.Input(shape=[512, 512, 2])
    

    down_stack = [
        downsample(64,  configuration['KERNEL_SIZE'],  configuration, apply_batchnorm=False),    
        downsample(64,  configuration['KERNEL_SIZE'],  configuration, apply_batchnorm=False),    
        downsample(128, configuration['KERNEL_SIZE'],  configuration),                          
        downsample(256, configuration['KERNEL_SIZE'],  configuration),                          
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                           
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                           
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                          
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                         
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                          
    ]

    up_stack = [
        upsample(512, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      
        upsample(512, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      
        upsample(512, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      
        upsample(512, configuration['KERNEL_SIZE'],    configuration),                          
        upsample(256, configuration['KERNEL_SIZE'],    configuration),                           
        upsample(128, configuration['KERNEL_SIZE'],    configuration),                          
        upsample(64,  configuration['KERNEL_SIZE'],    configuration),                           
        upsample(64,  configuration['KERNEL_SIZE'],    configuration)                            
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, configuration['KERNEL_SIZE'], strides=STRIDES, padding=PADDING, kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 512, 512, 1)
    #print("Last", last)

    x = inputs / 2
    
    # Downsampling through the model
    if configuration['USE_SKIP_CONNECTIONS']:
        skips = []
    for down in down_stack:
        x = down(x)
        # print("Downsample")
        # print(x)
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
            # print("Upsample")
            # print(x)

    x = last(x)
    # print('Kernel_size', configuration['KERNEL_SIZE'])
    # print("Last:" , x.shape)
    # print(inputs.shape)
    return tf.keras.Model(inputs=inputs, outputs=x)

def autoencoder_64x64(configuration):
    inputs = tf.keras.layers.Input(shape=[64, 64, 2])
    

    down_stack = [
        downsample(64,  configuration['KERNEL_SIZE'],  configuration, apply_batchnorm=False),    # (batch_size, 32, 32, 64)   
        downsample(128, configuration['KERNEL_SIZE'],  configuration),                           # (batch_size, 16, 16, 128)
        downsample(256, configuration['KERNEL_SIZE'],  configuration),                           # (batch_size, 8, 8, 256)
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                           # (batch_size, 4, 4, 512)
        downsample(512, configuration['KERNEL_SIZE'],  configuration),                          # (batch_size, 2, 2, 512)
        downsample(1024, configuration['KERNEL_SIZE'],  configuration),                          # (batch_size, 1, 1, 1024)
    ]

    up_stack = [
        upsample(512, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      # (batch_size, 2, 2, 512)
        upsample(512, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      # (batch_size, 4, 4, 512)
        upsample(256, configuration['KERNEL_SIZE'],    configuration, apply_dropout=True),      # (batch_size, 8, 8, 256)
        upsample(128, configuration['KERNEL_SIZE'],    configuration),                          # (batch_size, 16, 16, 128)
        upsample(64, configuration['KERNEL_SIZE'],    configuration),                           # (batch_size, 32, 32, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, configuration['KERNEL_SIZE'], strides=STRIDES, padding=PADDING, kernel_initializer=initializer, activation='sigmoid')  # (batch_size, 512, 512, 1)
    #print("Last", last)

    x = inputs / 2
    
    # Downsampling through the model
    if configuration['USE_SKIP_CONNECTIONS']:
        skips = []
    for down in down_stack:
        x = down(x)
        # print("Downsample")
        # print(x)
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
            # print("Upsample")
            # print(x)

    x = last(x)
    # print('Kernel_size', configuration['KERNEL_SIZE'])
    # print("Last:" , x.shape)
    # print(inputs.shape)
    return tf.keras.Model(inputs=inputs, outputs=x)



def Autoencoder(MODEL_NUMBER, configuration):
    if  MODEL_NUMBER==1:
        return autoencoder_split_256x256(configuration)
    elif MODEL_NUMBER==2:
        return autoencoder_256x256(configuration)
    elif MODEL_NUMBER==3:
        return autoencoder_512x512(configuration)
    elif MODEL_NUMBER==4:
        return autoencoder_64x64(configuration) # not finished
    else:
        raise ValueError("No_valid_Model_Number")
    
def masked_mse(image1, image2, mask):
    #print(f"image1,  {tf.reduce_min(image1)}, Maximum, {tf.reduce_max(image1)}, Mean, {tf.reduce_mean(image1)}, Shape: {tf.shape(image1)}")
    #print(f"image1,  {tf.reduce_min(image2)}, Maximum, {tf.reduce_max(image2)}, Mean, {tf.reduce_mean(image2)}, Shape: {tf.shape(image2)}")
    """Calculate MSE only for pixels within the mask"""
    #print(f"mask,  {tf.reduce_min(mask)}, Maximum, {tf.reduce_max(mask)}, Mean, {tf.reduce_mean(mask)}, Shape: {tf.shape(mask)}")
    masked_diff = (image1 - image2) * mask
    #print(f"masked_diff,  {tf.reduce_min(masked_diff)}, Maximum, {tf.reduce_max(masked_diff)}, Mean, {tf.reduce_mean(masked_diff)}, Shape: {tf.shape(masked_diff)}")
    #mse = tf.reduce_sum(tf.square(masked_diff)) / tf.reduce_sum(mask)
    mse = tf.reduce_sum(tf.square(masked_diff)) / tf.reduce_sum(mask)
    #print(f"minimum,  {mse}")
    return mse

def masked_mae(image1, image2, mask):
    """Calculate MAE only for pixels within the mask"""
    masked_diff = (image1 - image2) * mask
    mae = tf.reduce_sum(tf.abs(masked_diff)) / tf.reduce_sum(mask)
    return mae

def autoencoder_loss_calculation(autoencoder_output, target, configuration): # do this also
    if EVALUATION_METHOD == 0:
        l1_loss = tf.reduce_mean(pixelwise_distance(target, autoencoder_output, configuration))  # MAE (L1 loss)
        l2_loss = tf.reduce_mean(tf.square(pixelwise_distance(target, autoencoder_output, configuration)))  # MSE (L2 loss)     # Probelm pixelwise distance uses abs()
        
        total_autoencoder_loss = (l2_loss if configuration['USE_L2_LOSS'] else l1_loss) 
        return total_autoencoder_loss, l1_loss, l2_loss
    else:
        mask = get_mask(target)  # Generate the mask based on the target image
        #cv2.imwrite('training_mask.png', mask * 255)
        l1_loss = masked_mae(target, autoencoder_output, mask) 
        l2_loss = masked_mse(target, autoencoder_output, mask)  
        
        total_autoencoder_loss = (l2_loss if configuration['USE_L2_LOSS'] else l1_loss) # MAE = L1Loss MSE = L2Loss
        return total_autoencoder_loss, l1_loss, l2_loss

def generate_images(epoch, autoencoder, val_dataset_to_plot,training=True):

    img_idx = 0

    for batch in val_dataset_to_plot:
        
        nearfields, farfields, phasemasks = batch

        input_images = tf.concat([nearfields, farfields], axis=3)
        generated_images = autoencoder(input_images, training=training)  # TODO: Training True

        title = ['Input Nearfield', 'Input Farfield', 'Predicted Phasemask', 'Target Phasemask','Absolute Error']

        for images in zip(nearfields, farfields, generated_images, phasemasks):
            # preprocess the background after training.
            mask = get_mask(tf.expand_dims(images[3],axis=0))[0]
            gray_value = 128
            gray_value_noramlized = 128/255
            predicted_phasemask = images[2]
            adjusted_image = tf.where(mask[:,:,0] ==0, gray_value_noramlized,predicted_phasemask[:,:,0] )
            images =list(images)
            images[2] = adjusted_image
            
            images.append(tf.abs(images[2]-images[3][:,:,0]))

            # for image in images:
            #     #print(tf.shape(image))
            #     print()

            img_idx += 1
            images = list(images)

            plt.figure(figsize=(25, 5))
            for i in range(len(title)):
                plt.subplot(1, len(title), i+1)
                plt.title(title[i])
                plt.imshow(images[i], cmap='gray', interpolation='none')
                plt.axis('off')
            
            filename_prefix = 'epoch_' if str(epoch)[:17] !='final-performance' else ''
            if MODUS==0:
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/{filename_prefix}{epoch}__img_{img_idx}.png')
            else:
                pass
                #plt.savefig(r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\overview.png')
            
            plt.close('all')


            # Original Phasemask 
            if MODUS==0:
                plt.figure(figsize=(6.67, 6.67), dpi=300)
                plt.switch_backend('agg') # needed because of tkinder error
                plt.imshow(images[3], cmap='gray', interpolation='none')  
                plt.axis('off')
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/phasemask_img_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
            else:
                pass
                #save_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask.png'
                # image = images[3]
                # image_to_save = (image.numpy() * 255).astype(np.uint8)
                # image_original_size = resize(ORIGINAL_PHASEMASK_SIZE,image_to_save)
                # cv2.imwrite(save_path, image_original_size)

           
            if MODUS == 0:
                # predicted_phasemask = images[2]
                # #adjusted_image = predicted_phasemask * mask + (1 - mask) * gray_value
                # adjusted_image = tf.where(mask[:,:,0] ==0, gray_value_noramlized,predicted_phasemask[:,:,0] )
                #print(tf.shape(mask))
                #print(f"Shape preedicted mask {tf.shape(predicted_phasemask)}")
                #print(f"Shape mask {tf.shape(mask)}")
                #print(f"Adjusted {tf.shape(adjusted_image)}")
                

                # save image
                plt.figure(figsize=(6.67, 6.67), dpi=300)
                plt.switch_backend('agg') # needed because of tkinder error
                plt.imshow(adjusted_image, cmap='gray', interpolation='none')  
                plt.axis('off')
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/predicted_phasemask__{filename_prefix}{epoch}__img_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
            else:
                save_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask_predicted.png'
                image = images[2]
                image_to_save = (image.numpy() * 255).astype(np.uint8)
                image_original_size = resize(ORIGINAL_PHASEMASK_SIZE,image_to_save)
                cv2.imwrite(save_path, image_to_save)

            
            # Difference Image
            if MODUS==0:
                plt.figure(figsize=(6.67, 6.67), dpi=300)
                plt.switch_backend('agg') # needed because of tkinder error
                plt.imshow(images[4], cmap='gray', interpolation='none')  
                plt.axis('off')
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/difference_image__{filename_prefix}{epoch}__img_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
            else:
                save_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask_difference.png'
                image = images[4]
                image_to_save = (image.numpy() * 255).astype(np.uint8)
                image_original_size = resize(ORIGINAL_PHASEMASK_SIZE,image_to_save)
                cv2.imwrite(save_path, image_original_size)

            # Input Nearfield
            if MODUS==0:
                plt.figure(figsize=(6.67, 6.67), dpi=300)
                plt.switch_backend('agg') # needed because of tkinder error
                plt.imshow(images[0], cmap='gray', interpolation='none')  
                plt.axis('off')
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/nearfield_img_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
            else:
                pass

            # Input Farfield
            if MODUS==0:
                plt.figure(figsize=(6.67, 6.67), dpi=300)
                plt.switch_backend('agg') # needed because of tkinder error
                plt.imshow(images[1], cmap='gray', interpolation='none')  
                plt.axis('off')
                plt.savefig(f'{path_to_store_results}/{timestamp}/example_images/farfield_img_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.close('all')
            else:
                pass

                
            

def evaluate(autoencoder, val_dataset,configuration, training=True):
    autoencoder_loss_sum = 0
    l1_loss_sum = 0
    l2_loss_sum = 0
    ssim_sum = 0

    for batch in val_dataset:
        
        nearfields, farfields, phasemasks = batch
        input_images = tf.concat([nearfields, farfields], axis=3)

        generated_images = autoencoder(input_images, training=training)  # TODO: adapt?

        autoencoder_loss, l1_loss, l2_loss = autoencoder_loss_calculation(generated_images, phasemasks, configuration)

        
        # Calculate SSIM    
        if EVALUATION_METHOD == 0:
            ssim_score = float(tf.reduce_mean(tf.image.ssim(phasemasks, generated_images, max_val=1.0)))
        
        if EVALUATION_METHOD == 1:
            mask = get_mask(phasemasks)
            #cv2.imwrite('training_mask.png', mask * 255)
            ssim_score = float(tf.reduce_mean(ssim_specific(phasemasks, generated_images, mask, max_val=1.0 )))
            
        autoencoder_loss_sum += autoencoder_loss
        l1_loss_sum += l1_loss
        l2_loss_sum += l2_loss
        ssim_sum += ssim_score

    autoencoder_loss = float(autoencoder_loss_sum) / len(val_dataset)
    l1_loss = float(l1_loss_sum) / len(val_dataset)
    l2_loss = float(l2_loss_sum) / len(val_dataset)
    ssim_score = float(ssim_sum) / len(val_dataset)

    return  autoencoder_loss,l1_loss, l2_loss, ssim_score

def save_weights(epoch, autoencoder, delete_previous=False):
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
            pass  
    
    autoencoder.save_weights(f'{path_to_store_results}/{timestamp}/checkpoints/autoencoder/epoch_{epoch}')

#################################################################Training_Loop########################################################
if __name__ == "__main__":
    for config_idx, configuration in enumerate(configurations):

        if config_idx < SKIP_FIRST_N_CONFIGS:
            continue

        # create results folders
        timestamp = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '_')
        if MODUS == 0:
            os.mkdir(f'{path_to_store_results}/{timestamp}')
            os.mkdir(f'{path_to_store_results}/{timestamp}/example_images')
            os.mkdir(f'{path_to_store_results}/{timestamp}/history')

        # load data
        dataset = {}  
        for subset in ['train', 'test', 'val']:
            nearfield = folder_to_dataloader(f'{DATASET_PATH}/{subset}/beam_nf', configuration)
            farfield = folder_to_dataloader(f'{DATASET_PATH}/{subset}/beam_ff',configuration)
            phasemask = folder_to_dataloader(f'{DATASET_PATH}/{subset}/phasemask',configuration)
            dataset[subset] = tf.data.Dataset.zip((nearfield, farfield, phasemask))
        
        ######################################### create model #######################################################################
        # Loss Function
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Optimizer
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=configuration['LEARNING_RATE'], beta_1=configuration['BETA_1'], weight_decay= configuration['WEIGHT_DECAY'])
        # Model
        autoencoder = Autoencoder(MODEL_NUMBER, configuration)


        if PRETRAINED_WEIGHTS != '':
            autoencoder.load_weights(PRETRAINED_WEIGHTS.replace('MODEL', 'autoencoder'))

        ### DEFINE TRAINING ROUTINE

        @tf.function
        def train_step(input_images, targets, configuration):
            
            with tf.GradientTape() as gen_tape:
                
                autoencoder_output = autoencoder(input_images, training=True)
                autoencoder_loss, l1_loss, l2_loss = autoencoder_loss_calculation(autoencoder_output, targets, configuration)
        

                generator_gradients = gen_tape.gradient(autoencoder_loss, autoencoder.trainable_variables)
            

                generator_optimizer.apply_gradients(zip(generator_gradients, autoencoder.trainable_variables))


                try:
                    if EVALUATION_METHOD == 0:
                        ssim_score = float(tf.reduce_mean(tf.image.ssim(targets, autoencoder_output, max_val=1.0)))
                        return  autoencoder_loss, l1_loss, l2_loss, ssim_score
                    
                    if EVALUATION_METHOD == 1:
                        mask = get_mask(targets)

                        ssim_score = float(tf.reduce_mean(ssim_specific(targets, autoencoder_output,mask, max_val=1.0 )))
                        return  autoencoder_loss, l1_loss, l2_loss, ssim_score
                except:
                    print(f"This is not a valid case for the Hyperparameter EVALUATION_METHOD")



        def fit(train_dataset, val_dataset, n_epochs, configuration):

            history = {
                'epochs': {'train': [], 'val': []},
                'autoencoder_loss': {'train': [], 'val': []},
                'l1_loss': {'train': [], 'val': []},
                'l2_loss': {'train': [], 'val': []},
                'ssim': {'train': [], 'val': []}
            }

            val_dataset_to_plot = val_dataset.take(configuration['BATCHES_TO_PLOT'])
            consecutive_epochs_without_improvements = 0
            best_val_loss = float('inf')
            training_started_at = time.time()

            for epoch in range(1, n_epochs + 1):

                epoch_started_at = time.time()

                
                autoencoder_loss_sum = 0
                l1_loss_sum = 0
                l2_loss_sum = 0
                ssim_sum = 0
                
                for batch in train_dataset:
                    
                    nearfields, farfields, phasemasks = batch
                    input_images = tf.concat([nearfields, farfields], axis=3)
                    target_images = phasemasks

                    gen_gan_loss, l1_loss, l2_loss, ssim_score = train_step(input_images, target_images,configuration)

                    
                    autoencoder_loss_sum += gen_gan_loss
                    l1_loss_sum += l1_loss
                    l2_loss_sum += l2_loss
                    ssim_sum += ssim_score

                history['epochs']['train'].append(epoch)
                history['autoencoder_loss']['train'].append(float(autoencoder_loss_sum) / len(train_dataset))
                history['l1_loss']['train'].append(float(l1_loss_sum) / len(train_dataset))
                history['l2_loss']['train'].append(float(l2_loss_sum) / len(train_dataset))
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
                    generate_images(epoch, autoencoder, val_dataset_to_plot)

                if epoch % EVALUATE_EACH_N_EPOCHS == 0:
                    
                    autoencoder_loss, l1_loss, l2_loss,  ssim_score = evaluate(autoencoder, val_dataset, configuration)
                    
                    history['epochs']['val'].append(epoch)
                    history['autoencoder_loss']['val'].append(autoencoder_loss)
                    history['l1_loss']['val'].append(l1_loss)
                    history['l2_loss']['val'].append(l2_loss)
                    history['ssim']['val'].append(ssim_score)
                    log('\nCurrent performance:')
                    
                    for metric in history:
                        if metric == 'epochs':
                            continue
                        for subset in ['train', 'val']:
                            log(metric + '_' + subset + ' = ' + str(history[metric][subset][-1]))
                    log('')

                    # check for early stopping and save weights
                    if history['autoencoder_loss']['val'][-1] < best_val_loss:
                        best_val_loss = history['autoencoder_loss']['val'][-1]
                        consecutive_epochs_without_improvements = 0
                        save_weights(epoch, autoencoder, delete_previous=True)
                    else:
                        consecutive_epochs_without_improvements += EVALUATE_EACH_N_EPOCHS
                        if consecutive_epochs_without_improvements >= N_EPOCHS_WITHOUT_IMPROVEMENT_FOR_EARLY_STOPPING:
                            break

            save_weights(epoch, autoencoder, delete_previous=False)
                
            return history


        ### TRAIN MODEL

        history = fit(dataset['train'], dataset['val'], n_epochs=N_EPOCHS, configuration=configuration)
        log('history = ' + str(history) + '\n')


        ### EVALUATE MODEL

        # reset model to best-performing state
        best_idx_val = np.argmin(history['autoencoder_loss']['val'])
        best_epoch = history['epochs']['val'][best_idx_val]
        path_to_best_weights = f'{path_to_store_results}/{timestamp}/checkpoints/MODEL/epoch_{best_epoch}'
        autoencoder.load_weights(path_to_best_weights.replace('MODEL', 'autoencoder'))


        # analyze performance
        final_test_performance = {}
        final_test_performance_without_dropout ={}

        #Training True
        for subset in ['test']:
            autoencoder_loss, l1_loss, l2_loss, ssim_score = evaluate(autoencoder, dataset[subset],configuration)
            for metric in ['autoencoder_loss','l1_loss', 'l2_loss', 'ssim_score']:
                log(f'final {subset} {metric} = {eval(metric)}')
                final_test_performance[metric] = eval(metric)
            generate_images('final-performance', autoencoder, dataset[subset].take(configuration['BATCHES_TO_PLOT']))
        # Training=false
        for subset in ['test']:
            autoencoder_loss, l1_loss, l2_loss, ssim_score = evaluate(autoencoder, dataset[subset],configuration)
            for metric in ['autoencoder_loss','l1_loss', 'l2_loss', 'ssim_score']:
                log(f'final {subset} {metric} = {eval(metric)}')
                final_test_performance_without_dropout[metric] = eval(metric)
            generate_images('final-performance_without_dropout', autoencoder, dataset[subset].take(configuration['BATCHES_TO_PLOT']))


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
        plt.close('all')

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
            plt.close('all')
        
        
        ### LOGGING
        
        best_idx_train = best_epoch - 1
        
        # autoencoder_loss_train = history['autoencoder_loss']['train'][best_idx_train]
        # autoencoder_loss_val = history['autoencoder_loss']['val'][best_idx_val]
        # autoencoder_total_loss_test = final_test_performance['autoencoder_loss'] 
        # autoencoder_total_loss_test_without_dropout = final_test_performance_without_dropout['autoencoder_loss'] 
        l1_loss_train = history['l1_loss']['train'][best_idx_train]
        l1_loss_val = history['l1_loss']['val'][best_idx_val]
        l1_loss_test = final_test_performance['l1_loss']
        l1_loss_test_without_dropout = final_test_performance_without_dropout['l1_loss']
        l2_loss_train = history['l2_loss']['train'][best_idx_train]
        l2_loss_val = history['l2_loss']['val'][best_idx_val]
        l2_loss_test = final_test_performance['l2_loss']
        l2_loss_test_without_dropout = final_test_performance_without_dropout['l2_loss']
        ssim_train = history['ssim']['train'][best_idx_train]
        ssim_val = history['ssim']['val'][best_idx_val]
        with open(f'{path_to_store_results}/results_summary.csv', 'a') as f:
            f.write(f"{timestamp},{best_epoch},{l1_loss_train},{l1_loss_val},{l1_loss_test},{l1_loss_test_without_dropout},{l2_loss_train},{l2_loss_val},{l2_loss_test},{l2_loss_test_without_dropout},{ssim_train},{ssim_val},{final_test_performance['ssim_score']},{final_test_performance_without_dropout['ssim_score']}\n")
        

        # Safe the environment file
        content = (
        f"Python version: {sys.version} \n\n"
        f"pandas: {pd.__version__}\n"
        f"tensorflow: {tf.__version__}\n"
        f"matplotlib: {plt.matplotlib.__version__}\n"
        f"numpy: {np.__version__} \n\n"
        f"shutil: Built-in\n"
        f"math: Built-in\n"
        f"os: Built-in\n"
        f"time: Built-in\n"
        f"datetime: Built-in\n"
        )

        with open(f'{path_to_store_results}/python_libaries.csv', 'a') as f:
            f.write(content)



