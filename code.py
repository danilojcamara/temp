#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import scipy.signal
from scipy.signal import savgol_filter
%matplotlib qt

#%%
def crop_roi(img):
    return img[150:173, 341:350]

window_width = 5
reference = 1
limiar = 90   

#%%
cap = cv2.VideoCapture('video.AVI')
ret, frame = cap.read()
cv2.imwrite("frame.png", frame)
# cv2.imwrite("frame.png", crop_roi(frame))

#%%
cap = cv2.VideoCapture('video.AVI')
fps = cap.get(cv2.CAP_PROP_FPS)
window_width = 50
reference = 102

means = []
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    crop = crop_roi(frame)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = np.mean(grey)
    means.append(np.mean(grey))
df = pd.DataFrame({'mean': means})
df['meanhat'] = savgol_filter(df['mean'], 51, 3)
df['n'] = df.index.astype(int)
df['second'] = df['n'].apply(lambda x: x/fps)
df['time'] = df['second'].apply(lambda x: timedelta(seconds=x))
#%%
df
#%%
df.plot(x='second', y='meanhat')