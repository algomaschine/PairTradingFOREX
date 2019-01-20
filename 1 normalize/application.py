import pandas
from normalize import normalize
import sys

# Default parameters
type_ = "min-max"
q_lower = None
q_upper = None
ref = None
A = None
b = None

# User parameters
for i in range(len(sys.argv)):
    if sys.argv[i] == "-type":
        type_ = sys.argv[i+1]
    if sys.argv[i] == "-q_lower":
        q_lower = float(sys.argv[i+1])
    if sys.argv[i] == "-q_upper":
        q_upper = float(sys.argv[i+1])
    if sys.argv[i] == "-ref":
        ref = float(sys.argv[i+1])
    if sys.argv[i] == "-input":
        input_path = sys.argv[i+1]
    if sys.argv[i] == "-output":
        output_path = sys.argv[i+1]
    if sys.argv[i] == "-A":
        A = float(sys.argv[i+1])
    if sys.argv[i] == "-b":
        b = float(sys.argv[i+1])

# test
# input_path = r"E:\Machine_learning\task4\AUDNZDpro240.csv"
# output_path = r"E:\Machine_learning\task4\test2.csv"

# Sanity check
if input_path == None or output_path == None:
    sys.exit()

# import csv file as data frame
df = pandas.read_csv(input_path)
colnames = list(df.columns.values)

# convert data frame to a list of lists
list_ = df[colnames].values.tolist()
ls = []
for i in range(7):
    ls.append([])
for i in range(7):
    for j in range(len(list_)):
        ls[i].append(list_[j][i])

# Convert list to numberic values

if "Time" in colnames:
    for i in range(2, 7):
        ls[i] = list(map(float, ls[i]))
else:
    for i in range(1, 7):
        ls[i] = list(map(float, ls[i]))

# create output list
output = []
if "Time" not in colnames:
    for i in range(5):
        output.append(ls[i])
    normed_volume = [] # normalize volume column in list form
    normed_volume.append(0)
    for i in range(1, len(ls[5])):
        normed_volume.append( (ls[5][i] - ls[5][i-1]) / ls[5][i-1] )
    output.append(normed_volume)
    output.append(ls[6])
else:
    for i in range(6):
        output.append(ls[i])
    normed_volume = [] # normalize volume column in list form
    normed_volume.append(0)
    for i in range(1, len(ls[6])):
        normed_volume.append( (ls[6][i] - ls[6][i-1]) / ls[6][i-1] )
    output.append(normed_volume)

# convert back to data frame
output_df = pandas.DataFrame()
for i in range(len(colnames)):
    output_df[colnames[i]] = output[i]

# If A and b were not given, save x1 and x2 to calculate them
if A == None and b == None:
    x1 = output_df["Open"][0]
    x2 = output_df["Open"][1]

# normalize output
if "Time" in colnames:
    normcols = ["Open", "High", "Low", "Close"]
else:
    normcols = ['Open', 'High', 'Low', 'Close', 'Adj Close']

output_df[normcols] = normalize(output_df[normcols], type_=type_, q_lower=q_lower, q_upper=q_upper, ref=ref, A=A, b=b)

# Calculate A and b if not given
if A == None and b == None:
    y1 = output_df["Open"][0]
    y2 = output_df["Open"][1]
    A = (y1-y2) / (x1 - x2)
    b = y1 - A*x1
    print("A is {A}, and b is {b}".format(A=A, b=b))


# export data frame as csv file
output_df.to_csv(output_path, index=False)

