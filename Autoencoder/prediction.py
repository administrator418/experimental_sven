from timeit import default_timer


######################################################## Load Autoencoder and configs#######################################################
import os
#import time
#import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import yaml
from PIL import Image
import random

from Autoencoder_Architecture import autoencoder_512x512
from Autoencoder_Architecture import upsample    
from Autoencoder_Architecture import downsample
from Autoencoder_Architecture import Autoencoder
from Autoencoder_Architecture import evaluate
from Autoencoder_Architecture import generate_images
from Autoencoder_Architecture import folder_to_dataloader

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_hyperparameters(filepath):
    with open(filepath, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

filepath = 'config.yaml'
hyperparams = load_hyperparameters(filepath)

# needed config parameters
BEST_CONFIGURATION_ONLY = hyperparams['BEST_CONFIGURATION_ONLY']
MODUS= hyperparams['MODUS']
BEST_CONFIGURATION = hyperparams['BEST_CONFIGURATION']
N_EPOCHS = hyperparams['N_EPOCHS']  
MODEL_NUMBER = hyperparams['MODEL_NUMBER']


## updated this and the best config in the yaml file###################################################
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

##############################################################################################################################

# Define Datapaths
predict_data_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask'
FOLDER_PREDIRCTION = 'predict_512'


"""Load the model from local the time needed is 10 times higher when loading from the Cloud"""
# Load Model#
## simulated
# vortex simulated
#checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Simulation_Vortex_Abb08_best_config_MaxEpochs\2024-06-26_18-32-23-207937_fixed_shuffle_earlyStop10\2024-06-26_18-32-23-208132\checkpoints\autoencoder\epoch_48'
# gaussian simulated
# checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Simulation_Gaussian_Abb08_best config_MaxEpochs\2024-06-26_14-53-31-634438_fixed_shuffle_earlystop10\2024-06-26_14-53-31-634629\checkpoints\autoencoder\epoch_49'


# gaussian real ditzingen
# checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Real_Gaussian_Abb08_best_config_Max_Epochs\2024-07-06_00-00-37-904460\checkpoints\autoencoder\epoch_39'
# gaussian unwrapped
#checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Result_Gaussian_unwrapped_EpochsMax\2024-07-22_10-35-05-443658\checkpoints\autoencoder\epoch_32'
# vortex real ditzingen
#checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Real_Vortex_Abb08_best_config_Max_Epochs\2024-07-05_05-39-55-532277\checkpoints\autoencoder\epoch_38'
# vortex real ditzingen v2
#checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\Real_vortex_Abb08_best_config_Max_Epochs_v2\2024-07-17_08-38-11-746144\2024-07-17_08-38-11-746325\checkpoints\autoencoder\epoch_34'
#checkpoint_path = r'C:\Results\Real_vortex_Abb08_best_config_Max_Epochs_v2\2024-07-17_08-38-11-746144\2024-07-17_08-38-11-746325\checkpoints\autoencoder\epoch_34'

## Unwrapped
# vortex unwrapped
#checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\unwrapped\Ditzingen_Vortex_Unwrapped_Max_Epochs\2024-08-01_11-18-31-680771\checkpoints\autoencoder\epoch_50'
# gaussian unwrapped
checkpoint_path = r'\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Results\Final_Ditzingen\unwrapped\Ditzingen_Gaussian_Unwrapped_Max_Epochs\2024-08-05_15-12-22-276854\checkpoints\autoencoder\epoch_34'



##  Schramberg
#checkpoint_path = r'C:\Users\burckhardsv\Lokale_Dateien\Schramberg\Schramberg_Datacollection\Schramberg_results\result_vortex_config22\2024-09-05_08-25-38-836052\checkpoints\autoencoder\epoch_40'
#checkpoint_path = r'C:\Users\burckhardsv\Lokale_Dateien\Schramberg\Schramberg_Datacollection\Schramberg_results\result_vortex_config22_circle\2024-09-05_13-07-52-962704\checkpoints\autoencoder\epoch_40'




# Load model with the weights
for config_idx, configuration in enumerate(configurations):
    print(configuration)
    model = Autoencoder(MODEL_NUMBER=MODEL_NUMBER, configuration=configuration)


model.load_weights(checkpoint_path)



# Prediction
def predict(predict_data, autoencoder,configuration):
    """Predict the Phasemask and give out all metrics and the overview image"""
    final_test_performance = {}
    for subset in [FOLDER_PREDIRCTION]:
        #start_timer = default_timer()  
        autoencoder_loss, l1_loss, l2_loss, ssim_score = evaluate(autoencoder, predict_data[subset],configuration)
        #duration = default_timer() - start_timer
        #print(f"Time: {duration}")
        for metric in ['autoencoder_loss','l1_loss', 'l2_loss', 'ssim_score']:
            final_test_performance[metric] = eval(metric)
        start_timer = default_timer()    
        generate_images('final-performance', autoencoder, predict_data[subset].take(configuration['BATCHES_TO_PLOT']))
    return autoencoder_loss, l1_loss, l2_loss, ssim_score

predict_data = {}  
for subset in [FOLDER_PREDIRCTION]: # Folder with the subfolders NF/FF/PM
    nearfield = folder_to_dataloader(f'{predict_data_path}/{subset}/beam_nf',configuration)
    farfield = folder_to_dataloader(f'{predict_data_path}/{subset}/beam_ff',configuration)
    phasemask = folder_to_dataloader(f'{predict_data_path}/{subset}/phasemask',configuration)
    predict_data[subset] = tf.data.Dataset.zip((nearfield, farfield, phasemask))



autoencoder_loss, l1_loss, l2_loss, ssim_score = predict(predict_data, model,configuration)


print(f"MAE: {l1_loss} | MSE: {l2_loss}| SSIM: {ssim_score}")

def reverse_phasemask(image_path):
    img = Image.open(image_path)
    inverted_img = Image.eval(img, lambda px: 255 - px)  # Invert pixel values
    return inverted_img

image_path = r"\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask_predicted.png"
# inverted_image = reverse_phasemask(image_path)
# output_path = r"\\srvditz1\lac\Studenten\AE_VoE_Stud\Sven Burckhard\Predict_Phasemask\phasemask_predicted_reverse.png"  # Passe den Pfad an
# inverted_image.save(output_path)

print("Finished")
