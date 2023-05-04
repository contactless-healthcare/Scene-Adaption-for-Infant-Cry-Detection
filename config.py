# Set data
audioSampleRate = 8000
slidingWindows = 2.5
overlap = 2.5


# Feature Extraction
FFTwindow = 256
FFTOverlap = 128
MFCCFiliterNum = 26


# Model Design
dropout = 0.4


# Train Setting
Data_Augmentation_Bool = True
batch_size = 64
lr = 0.0001
weight_decay = 0.0001
confidence_threshold = 0.7


SL_EPOCH = 50   # the epoch trained on CSD
SA_EPOCH = 20   # scene adaption epoch





