import math
import sys

import numpy as np
import pandas as pd

# variables for test purposes
# input1 = r"E:\Machine_learning\task4\test.csv"
# input2 = r"E:\Machine_learning\task4\test2.csv"
# output = r"E:\Machine_learning\task4\test_result.csv"
# minDX = 0.01
# minDT = 5

# User parameters
for i in range(len(sys.argv)):
    if sys.argv[i] == "-input1":
        input1 = sys.argv[i+1]
    if sys.argv[i] == "-input2":
        input2 = (sys.argv[i+1])
    if sys.argv[i] == "-output":
        output = (sys.argv[i+1])
    if sys.argv[i] == "-minDX":
        minDX = float(sys.argv[i+1])
    if sys.argv[i] == "-minDT":
        minDT = int(sys.argv[i+1])

# Read input csv files and concatenate them
df1 = pd.read_csv(input1)
df2 = pd.read_csv(input2)
df_merged = pd.concat([df1, df2], axis=1)

# Drop the second date column
if "Adj Close" in list(df_merged.columns.values):
    df_merged.columns = ['Date', 'Open1', 'High1', 'Low1', 'Close1', 'Volume1', 'Adj Close1',
                         'remove', 'Open2', 'High2', 'Low2', 'Close2', 'Volume2', 'Adj Close2']
else:
    df_merged.columns = ['Date', 'Time1', 'Open1', 'High1', 'Low1', 'Close1', 'Volume1',
                         'remove', 'Time2', 'Open2', 'High2', 'Low2', 'Close2', 'Volume2']
df_merged = df_merged.drop('remove', axis=1)

# Compute the difference between Close1 and Close2
diff = df_merged["Close1"] - df_merged["Close2"]
length = len(diff)

# Initialize list for signal and intersection points
signal = [None] * length
inter_points = []

# Find intersection points and set their signal to 0
for i in range(length):
    if abs(diff[i]) < minDX:
        signal[i] = 0
        inter_points.append(i)

# Set signals within minDT buffer from intersection points to 0. Check for intersections at the ends
for point in inter_points:
    if minDT >= point and point < length - minDT:
        for i in range(point + minDT + 1):
            signal[i] = 0
    if minDT >= point >= length - minDT:
        for i in range(length):
            signal[i] = 0
    if minDT < point < length - minDT:
        for i in range(point - minDT, point + minDT + 1):
            signal[i] = 0
    if minDT < point and point >= length - minDT:
        for i in range(point - minDT, length):
            signal[i] = 0

# check for cases where the diff function jumps over 0
for i in range(length - 1):
    if (diff[i] < -minDX and diff[i+1] > minDX) or (diff[i] > -minDX and diff[i+1] < minDX):
        inter_points.append(i + 0.5)
        inter_points.sort()

# Calculate interval minimums and maximums
interval_min_diff = []
interval_max_diff = []
for i in range(len(inter_points) - 1):
    interval_min_diff.append(np.min(diff[math.ceil(inter_points[i]): math.floor(inter_points[i+1]):]))
    interval_max_diff.append(np.max(diff[math.ceil(inter_points[i]): math.floor(inter_points[i+1]):]))

# Check whether diff function on intervals with non-zero values is positive or negative
# for i in range(len(inter_points)-1):
    # if math.ceil(inter_points[i+1]) - math.floor(inter_points[i]) > 2 * minDT + 2:
        # if (diff[math.floor(inter_points[i])] + diff[math.ceil(inter_points[i+1])]) // 2 > 0:
            # interval_min_diff[i] = 0
        # elif (diff[math.floor(inter_points[i])] + diff[math.ceil(inter_points[i+1])]) // 2 < 0:
            # interval_max_diff[i] = 0

# Calculate signals out of buffer range
for i in range(length):
    for j in range(len(inter_points) - 1):
        if signal[i] is None and math.ceil(inter_points[j]) < i < math.floor(inter_points[j + 1]):
            if interval_max_diff[j] > minDX:
                signal[i] = diff[i] / interval_max_diff[j]
            elif interval_min_diff[j] < -minDX:
                signal[i] = - diff[i] / interval_min_diff[j]
            else:
                signal[i] = 0

# Set signals before first intersection to 5, the ones after the last to -5
for i in range(math.ceil(inter_points[0])):
    signal[i] = 5
for i in range(math.ceil(inter_points[len(inter_points) - 1] + 1), length):
    signal[i] = -5

# Concatenate signal column to the data frame
signal_df = pd.Series(signal)
df_merged['Signal'] = signal_df.values

# export data frame as csv file
df_merged.to_csv(output, index=False)
