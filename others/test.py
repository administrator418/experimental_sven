import tensorflow as tf

# 检查是否有可用的 M1 加速器
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
