How to use:
	- Required arguments:
		-input : input csv path
		-output : output csv path
	- Optional arguments:
		-type: "min-max", "quantile", "ref", "Ab"
			if no -type argument is given, it uses "min-max" normalization as standard
		-q_lower: lower quantile for type "quantile" normalization between 0 and 1
		-q_upper: upper quantile for type "quantile" normalization between 0 and 1
		-ref: reference value for type "ref" normalization
		-A : A value for type "Ab" normalization
		-b : b value for type "Ab" normalization

Example command line arguments:
application.py -input "E:\Machine_learning\task2\table.csv" -output "E:\Machine_learning\task2\test.csv"
application.py -input "E:\Machine_learning\task2\table.csv" -output "E:\Machine_learning\task2\test.csv" -type min-max        # does the same as the line before
application.py -input "E:\Machine_learning\task2\table.csv" -output "E:\Machine_learning\task2\test.csv" -type quantile -q_lower 0.25 -q_upper 0.75
application.py -input "E:\Machine_learning\task2\table.csv" -output "E:\Machine_learning\task2\test.csv" -type ref -ref 80
application.py -input "E:\Machine_learning\task2\table.csv" -output "E:\Machine_learning\task2\test.csv" -type Ab -A 0.017314638441174052 -b -0.8138815577266735