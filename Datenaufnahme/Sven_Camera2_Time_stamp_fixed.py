# written by Sven Burckhard
"""Careful, safe the Images only local"""
print("start camera_2")
from pyueye import ueye
import numpy as np
import cv2
import sys
import os
from PIL import Image
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer
import pandas as pd
import csv

# Variables
hCam = ueye.HIDS(2)  # 0: first available camera;  1-254: The camera with the specified camera ID
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
pcImageMemory = ueye.c_mem_p()
MemID = ueye.int()
rectAOI = ueye.IS_RECT()
pitch = ueye.INT()
nBitsPerPixel = ueye.INT(8) #before[24]  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
channels = 1 # before[3]  # 3: channels for color mode(RGB); take 1 channel for monochrome
m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
bytes_per_pixel = int(nBitsPerPixel / 8)


# Starts the driver and establishes the connection to the camera
nRet = ueye.is_InitCamera(hCam, None)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_InitCamera ERROR Camera 2")

# Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
nRet = ueye.is_GetCameraInfo(hCam, cInfo)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_GetCameraInfo ERROR")

# You can query additional information about the sensor type used in the camera
nRet = ueye.is_GetSensorInfo(hCam, sInfo)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_GetSensorInfo ERROR")

nRet = ueye.is_ResetToDefault(hCam)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_ResetToDefault ERROR")

# Set display mode to DIB
nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_SetDisplayMode ERROR")

# Set the right color mode
if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
    ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("IS_COLORMODE_BAYER: ", )
#         print("\tm_nColorMode: \t\t", m_nColorMode)
#         print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
#         print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
#         print()
elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
    m_nColorMode = ueye.IS_CM_BGRA8_PACKED
    nBitsPerPixel = ueye.INT(32)
    bytes_per_pixel = int(nBitsPerPixel / 8)
#         print("IS_COLORMODE_CBYCRY: ", )
#         print("\tm_nColorMode: \t\t", m_nColorMode)
#         print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
#         print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
#         print()
elif int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
else:
    m_nColorMode = ueye.IS_CM_MONO8
    nBitsPerPixel = ueye.INT(8)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    print("else")

# Can be used to set the size and position of an "area of interest"(AOI) within an image
nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_AOI ERROR")

width = rectAOI.s32Width
height = rectAOI.s32Height

# Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_AllocImageMem ERROR")

# Makes the specified image memory the active memory
nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_SetImageMem ERROR")

# Set the desired color mode
nRet = ueye.is_SetColorMode(hCam, m_nColorMode)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_SetColorMode ERROR")

# Activates the camera's live video mode (free run mode)
nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_CaptureVideo ERROR")

# Enables the queue mode for existing image memory sequences
nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_InquireImageMem ERROR")

# Disable auto gain
enable = ueye.double(0)
dummy = ueye.double(0)
nRet = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, enable, dummy)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_SetAutoParameter for AUTO_GAIN ERROR")

# Set fixed gain value
fixed_gain = 0  # Adjust this value as necessary
nRet = ueye.is_SetHardwareGain(hCam, fixed_gain, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER, ueye.IS_IGNORE_PARAMETER)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_SetHardwareGain ERROR")

# Disable auto exposure
nRet = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable, dummy)
if nRet != ueye.IS_SUCCESS:
    raise ValueError("Error disabling auto exposure")

# HERE YOU CAN ADJUST THE CAMERA EXPOSRUE TIME 
time_exposure_ = 32 #30

time_exposure = ueye.double(time_exposure_)
nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))
if nRet != ueye.IS_SUCCESS:
    raise ValueError("is_Exposure ERROR")

def generate_timestamps(start_time_str, count, interval_seconds):
    # Startzeit im Format HH:MM:SS
    start_time = datetime.strptime(start_time_str, "%H:%M:%S")
    timestamps = []

    for i in range(count):
        # Berechne den neuen Zeitstempel
        new_timestamp = start_time + timedelta(seconds=i * interval_seconds)
        # Füge den neuen Zeitstempel zur Liste hinzu im Format HH:MM:SS
        timestamps.append(new_timestamp.strftime("%H:%M:%S"))

    return timestamps


# Settings
num = 2 # here to define how many images you want to collect  TODO: First image black second image fine.
z=199# number of images
start_time = "12:04:00"
start_time_camera_2 = (datetime.strptime(start_time, "%H:%M:%S") + timedelta(seconds=2)).strftime("%H:%M:%S")
interval_seconds = 4

timestamps = generate_timestamps(start_time_camera_2, z, interval_seconds)

# # timestamps in csv speichern
# csv_filename = "timestamps_camera_1.csv"

# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Timestamp"])  # Header schreiben
#     for ts in timestamps:
#         writer.writerow([ts])

for i in range(z):
    print(i)
    timestamp = timestamps[i]
    now = datetime.now()
    target_time = datetime.combine(now.date(), datetime.strptime(timestamp, "%H:%M:%S").time())

    # Adjust for the case where the target time is on the next day
    if target_time < now:
        target_time += timedelta(days=1)

     # Wait until the target timestamp is reached
    while datetime.now() < target_time:
        pass

    for j in range(1):  # Für jedes z nur ein Bild 
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        str_current_datetime = str(current_datetime)
        file_name = f"{i+1}_idx_{str_current_datetime}_c2_1.png"
        
        # Define the path for saving images and the CSV file
        #csv_path = r'\\srvditz1\HOME$\01_TLD712_photron\Desktop\Sven_Burckhard\save_image_time_camera_2.csv'
        #csv_path = r'\\srvditz1\home$\burckhardsv\Desktop\Datenaufnahme\save_image_time_camera_2.csv'

        # Initialize the CSV file with headers if it doesn't exist
        # if not os.path.exists(csv_path):
        #     with open(csv_path, mode='w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(['Iteration', 'Total Time (s)', 'Raw Image Time (s)'])  
        
        for i in range(num):
            # Get raw image
            # loop_start_time = time.time()
            # raw_image_start_time = time.time()
            raw_image = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
            # raw_image_time = time.time() - raw_image_start_time

            if raw_image is None:
                print("Getting image failed.")
                continue
            
            raw_image = np.reshape(raw_image,(2048, 2048))
            # Show acquired image
            img = Image.fromarray(raw_image, 'L')
            #path_camera_1 = r'C:\Sven\Schramberg_Datacollection\camera_2'
            #path_camera_1 = r'C:\Sven\Schramberg_Datacollection\reference_camera_2'
            path_camera_1 = r'C:\Sven\Schramberg_Datacollection\Correction_NF_FF\camera_2'
            full_path = os.path.join(path_camera_1, file_name)
            img.save(full_path)
            time.sleep(1)  # before 3

            #total_time = time.time() - loop_start_time

            # #Append the times to the CSV file
            # with open(csv_path, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([i, total_time, raw_image_time])
            # print("test")

print("Finished Data Collection")

if pcImageMemory and MemID:
        nRet = ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_FreeImageMem ERROR")

# Disable the camera handle
if hCam:
    nRet = ueye.is_ExitCamera(hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ExitCamera ERROR Camera 2")

print("Camera 2 closed")