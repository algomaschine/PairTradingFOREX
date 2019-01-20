from functions import create_df, save_plot, sanity_checks, LSTM_model,\
    create_matrixes, arg_parser, save_model, load_model, train_model, predict_values

# Parse command line arguments
args = arg_parser()

# Sanity checks
sanity_checks(args)

# Train neural network if --table is given
if args.table is not None:
    # Prepare data
    df = create_df(args)

    # Create input and output matrices
    input_mat, target_mat = create_matrixes(df, args.window, args.pred_len)

    # Set up new model, if no loadfile is given, else load it
    if args.load is None:
        model = LSTM_model(input_mat, args.layers, args.window, args.dropout, args.pred_len)
    else:
        model = load_model(args, df.columns.values)

    # Train model
    history, fit = train_model(model, input_mat, target_mat, args)

    # Save plot of loss history
    if args.plot is not None:
        save_plot(fit, args.plot)

    # Save model
    if args.save is not None:
        save_model(model, args, df.columns.values)

# Run the neural network being trained on through the network if --validate is given
if args.validate is not None:
    predict_values(args, args.table, args.validate, args.save, validation=True)

# Predict signal values if --pred_inp is given
# If network is trained load from --save, otherwise from --load
if args.pred_inp is not None:
    if args.save is not None:
        predict_values(args, args.pred_inp, args.pred_out, args.save, validation=False)
    else:
        predict_values(args, args.pred_inp, args.pred_out, args.load, validation=False)
