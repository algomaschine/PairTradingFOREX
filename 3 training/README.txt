NEURAL NETWORK training and prediction scripts

Prerequisites:

- Python 3.5
- Numpy, Scipy, pandas, matplotlib
- Keras, Tensorflow
- OpenBLAS environment (easy on Linux, Mac) or Intel python package (works for me on Windows). 

Usage:
1. Training from scratch
python application.py -t <input_file.csv> --layers <NN architecture> --window <window_size> -s <model_name.h5>

Input:
-t: 			Training data, output of the signal generation script. Must contain "Signal" column.
--layers: 		Number and size of hidden layers. E.g., --layers 100 200 specifies a network with 2 hidden layers, one with 100 nodes and one with 200 nodes.
--window: 		How many steps to look back for prediction?
--pred_len:		How many steps ahead to predict?
-s: 			Output model file name

Optional parameters:
-i, --ignore: 	Columns to be ignored
--testset:		Fraction of the input NOT used for training, but for validation only. Can be set to 0 in real applications.
--dropout:		Training parameter, helps against overfitting
--batch:		Training parameter, responsible mainly for speed
--epoch:		Training parameter, number of iterations
--plot:			Save training results to a figure
-v,--validate:	File name for saving prediction results for the training file


2. Training an existing model using new data (update)

python application.py -t <input_file.csv> -l <model_name_in.h5> -s <model_name_out.h5>

Input:
-l:				Model to load
-t, -s:			Same as above. the argument of -s may be the same as of -l


Optional parameters:
--dropout, --testset, --batch, --epoch, -v:	Same as above

3. Applying a model to make predictions

python application.py --pred_inp <input_file.csv> --pred_out <output_file.csv>  -l <model_name_in.h5>

Input:
--pred_inp:		Input file. The "Signal" column is optional
--pred_out:		Output file with predictions.
