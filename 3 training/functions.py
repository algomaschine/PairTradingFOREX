import argparse
import csv
import os
import sys

import keras
import numpy as np
import pandas as pd
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt

from classes import LossHistory


def arg_parser():
    """
    parses user input
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', '-s', default=None,
                        help="Save model as hdf5 file.")
    parser.add_argument('--load', '-l', default=None,
                        help="Load model from a hdf5 file.")
    parser.add_argument('--table', '-t',
                        help='Input csv file for training. Must contain "Signal" column.')                      
    parser.add_argument('--validate', '-v',
                        help="Output file (.csv) with the predictions for the training input (specified using --table) ")

    parser.add_argument('--pred_inp',
                        help='Input csv file for which signal values must be predicted.')
    parser.add_argument('--pred_out',
                        help="the location of the prediction results to be saved to. (.csv)")
    
    parser.add_argument('--window', type=int, default=100,
                        help="Number of days to look back on. Default: 100")
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help="The layer structure, with each number after the -layer argument representing" +
                             ' an LSTM layer with a number of units. E.g.: "--layers 50 100 50"')
    parser.add_argument('--ignore', '-i', action='append', default=None,
                        choices=["Open1", "High1", "Low1", "Close1", "Volume1", "Adj Close1",
                                 "Open2", "High2", "Low2", "Close2", "Volume2", "Adj Close2"],
                        help='Name of the columns to be ignored. Can be used more than once.')          
    parser.add_argument('--pred_len', type=int, default=1,
                        help="Number of days to predict. Default: 1")
    parser.add_argument('--batch', type=int, default=128,
                        help="The size of each batch. Default: 128")
    parser.add_argument('--epoch', type=int, default=3,
                        help="The number of epochs to run. Default: 3")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Rate of dropout. Number between 0 and 1. Default: 0.2")
    parser.add_argument('--testset', type=float, default=0.2,
                        help="The size of the test set. Number between 0 and 1. " +
                             "Set to 0 to train on the whole table. " +
                             "Set to 1 to validate on the whole table. Default: 0.2")
   
    parser.add_argument('--plot', default=None,
                        help="Destination of the png plot of the loss history (training success monitor).")

    # Display help if no arguments were given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

######################################################


def sanity_checks(args):
    """
    Checks whether input and output files are valid
    :return:
    """

    if args.table is None and args.pred_inp is None:
        sys.exit("Please enter an input or prediction table. (--table or --pred_inp)")

    if args.table is not None:
        if not os.path.isfile(args.table):
            sys.exit("Input table does not exist. (--table)")
        if os.path.splitext(args.table)[1] != ".csv":
            sys.exit("Input table is not a csv file. (--table)")
        if args.save is None:
            sys.exit("No training without saving the model! (--table, --save ) ")

    if args.pred_inp is not None:
        if not os.path.isfile(args.pred_inp):
            sys.exit("Input prediction table does not exist. (--pred_inp)")
        if os.path.splitext(args.pred_inp)[1] != ".csv":
            sys.exit("Input prediction table is not a csv file. (--pred_inp)")

    if args.pred_out is not None:
        if os.path.splitext(args.pred_out)[1] != ".csv":
            sys.exit("Output prediction is not a csv file. (--pred_out)")

    if args.validate is not None:
        if os.path.splitext(args.validate)[1] != ".csv":
            sys.exit("Validation output is not a csv file. (--validate)")

    if args.pred_inp is not None and args.load is None and args.table is None:
        sys.exit("Prediction requires a new or a loaded model. (--table or --load)")
    if args.pred_inp is not None and args.pred_out is None:
        sys.exit("Please enter a location for the prediction to be saved. (--pred_out)")

    if args.validate is not None and args.table is None:
        sys.exit("Validation(--validate) requires input table. (--table)")

    if args.save is not None:
        if os.path.splitext(args.save)[1] != ".h5":
            sys.exit("Please save model in .h5 format! (--load)")

    if args.load is not None:
        if not os.path.isfile(args.load):
            sys.exit("Invalid load file. (--load)")
        if os.path.splitext(args.load)[1] != ".h5":
            sys.exit("Please load an h5 file! (--load)")
        load_csv_name = os.path.splitext(args.load)[0] + ".csv"
        if not os.path.isfile(load_csv_name):
            sys.exit("Missing csv load file! ({})".format(load_csv_name))
        if args.ignore is not None:
            sys.exit("When loading a model, can not select which columns to ignore. (--ignore)")
        if args.layers is not None:
            sys.exit("When loading a model, can not change layer structure. (--layers)")

    if args.load is None and args.layers is None:
        sys.exit("Please input layer structure. --layer")

    if args.pred_out is not None and args.pred_inp is None:
        sys.exit("Please select an input table for prediction. (--pred_inp)")

    if args.plot is not None:
        if os.path.splitext(args.plot)[1] != ".png":
            sys.exit("Please save the plot in .png format. (--plot)")

###############################################


def create_df(args):
    """
    Creates an appropriate dataframe for the stocks table
    :return: pandas dataframe
    """

    # load the dataframe, drop unneeded columns, then create dataset
    df = pd.read_csv(args.table)

    # drop unneeded columns from --ignore if --load is not given
    if args.load is None:
        cols_to_drop_from_user = args.ignore
        if cols_to_drop_from_user is None:
            cols_to_drop_from_user = []
        if "Date" in list(df.columns.values):
            cols_to_drop_from_user.append('Date')
        if "Time1" in list(df.columns.values):
            cols_to_drop_from_user.append("Time1")
            cols_to_drop_from_user.append("Time2")
        df = df.drop(cols_to_drop_from_user, axis=1)

    else:
        load_name = os.path.splitext(args.load)[0] + ".csv"
        cols_used = pd.read_csv(load_name, header=None)
        cols_used = cols_used.values[0]
        df = df[cols_used]

    # drop unneeded rows
    df = df.drop(df[df.Signal == 5].index)
    df = df.drop(df[df.Signal == -5].index)

    # replace missing values with 0s
    nr_of_rows_with_na = df.shape[0] - df.dropna().shape[0]
    print("The number of rows containing missing values is {}".format(nr_of_rows_with_na))
    df = df.fillna(0)

    return df


########################################################

def create_matrixes(df, window, prediction):
    """
    Creates input and output matrices for the model to read
    :param df: input table in dataframe format
    :param window: number of days to look back on
    :param prediction: number of values to predict
    :return: an input matrix and an output matrix
    """
    data = df.values
    nb_samples = len(data) - window - prediction

    # input
    input_cols = df.columns.values[:-1:]
    input_data = df[input_cols].values
    input_list = [np.expand_dims(np.atleast_2d(input_data[i:window + i, :]), axis=0) for i in range(nb_samples)]
    input_mat = np.concatenate(input_list, axis=0)

    # target
    y = len(df.columns.values) - 1  # index of the signal column
    target_list = [np.atleast_2d(data[i + window:window + i + prediction, y]) for i in range(nb_samples)]
    target_mat = np.concatenate(target_list, axis=0)

    return input_mat, target_mat


##################################################

def LSTM_model(input_mat, layers, window, dropout_rate, prediction):
    """
    LSTM model structure
    :param input_mat: input matrix
    :param layers: an array, with each value representing a layer with units
    :param window: number of days to look back on
    :param dropout_rate: dropout rate
    :param prediction: number of values to predict
    :return:
    """

    model = Sequential()

    features = input_mat.shape[2]

    model.add(LSTM(layers[0], input_shape=(window, features), return_sequences=True))
    model.add(Dropout(dropout_rate))

    for i in range(1, len(layers) - 1):
        model.add(LSTM(
            layers[i],
            return_sequences=True
        ))
        model.add(Dropout(dropout_rate))

    model.add(LSTM(
        layers[len(layers) - 1],
        return_sequences=False
    ))
    model.add(Dropout(dropout_rate))

    model.add(Dense(prediction))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model


##################################################

def load_model(args, columns_new):
    """
    load existing model and cross-check ignored columns
    :param args:
    :param columns_new:
    :return:
    """
    model = keras.models.load_model(args.load)
    load_name = os.path.splitext(args.load)[0] + ".csv"
    col_check = pd.read_csv(load_name, header=None)
    columns_loaded = col_check.values[0]

    if not np.array_equal(columns_loaded, columns_new):
        sys.exit("Columns of loaded and input tables are not the same! \n" +
                 "Columns of the loaded table: {}. \n".format(columns_loaded) +
                 "Columns of the new table: {}".format(columns_new))

    return model

##################################################


def train_model(model, input_mat, target_mat, args):
    """
    Train neural network
    :param model: model to train
    :param input_mat: input in matrix format
    :param target_mat: target in matrix format
    :param args: input arguments
    :return: history of the model and fitted model
    """
    history = LossHistory()
    model.summary()
    fit = model.fit(input_mat, target_mat, epochs=args.epoch,
                    batch_size=args.batch, callbacks=[history], validation_split=args.testset)

    if args.save is None:
        print("Warning: model has not been saved. (--save, -s)")

    return history, fit

##################################################


def save_plot(model, plot_loc):
    """
    Saves a plot of the history of the loss function
    :param model: model to plot
    :param plot_loc: where to save the plot
    :return:
    """

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(plot_loc)
    plt.close()

    print("Plot has been saved to {}".format(plot_loc))


###########################################


def save_model(model, args, columns):
    """
    Saves weights, ignored columns, window size
    :return:
    """

    # Save model to given location
    model.save(args.save)

    # Save column names to a csv file
    save_name = os.path.splitext(args.save)[0]
    save_name += ".csv"
    with open(save_name, "w") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(columns)  # column names
    print("Model has been saved to {} and {}.".format(args.save, save_name))

############################################


def predict_values(args, input, output, load, validation):
    """
    Takes a csv table and predicts the signal column
    :param args: user input
    :param input: Table with signal column to be predicted (csv)
    :param output: Table with signal column predicted (csv)
    :param load: model (h5 and csv)
    :param validation: boolean
    :return: csv
    """

    print("Preparing output table...")

    model = keras.models.load_model(load)

    # Get columns needed from load csv file
    load_name = os.path.splitext(load)[0] + ".csv"
    cols_used = pd.read_csv(load_name, header=None)
    cols_used = cols_used.values[0][:-1:]

    # Create matrix for input prediction
    df = pd.read_csv(input)
    df = df.fillna(0)
    prediction_input = df[cols_used].values
    window_size = model.input_shape[1]
    nb_samples = len(prediction_input) - window_size
    input_list = [np.expand_dims(np.atleast_2d(prediction_input[i:window_size + i, :]), axis=0) for i in range(nb_samples)]
    input_mat = np.concatenate(input_list, axis=0)

    # Predict values
    prediction = model.predict([input_mat], batch_size=args.batch, verbose=0)

    # Fill prediction value data frame so it matches up with prediction input data frame
    prediction_length = model.output_shape[1]
    prediction_df = pd.DataFrame(prediction)
    fill = pd.DataFrame(index=range(window_size), columns=range(prediction_length))
    fill = fill.fillna(5)
    prediction_df = pd.concat([fill, prediction_df], ignore_index=True)

    # Rename columns
    pred_column_names = []
    for i in range(prediction_length):
        pred_column_names.append("Prediction + {}".format(i + 1))
    prediction_df.columns = pred_column_names

    # Concatenate prediction input and value into one csv, then export it
    result = pd.concat([df, prediction_df], axis=1)
    result.to_csv(output)

    if validation is False:
        print("Prediction table with predicted values has been exported to {}.".format(output))
    else:
        print("Validation table with predicted values has been exported to {}.".format(output))












