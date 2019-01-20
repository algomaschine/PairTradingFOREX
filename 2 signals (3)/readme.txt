How to use:
	- Required arguments:
		-input1: path to first csv file
		-input2: path to second csv file
		-output: path to output csv file
		-minDX: min difference between normalized rates, which may yield a non-zero signal 
		-minDT: within this number of ticks from the intersection points of the two normalized graphs the signal is set to 0

Example call:
application.py -input1 .\test1.csv -input2 .\test2.csv -output .\test_result.csv -minDX 0.01 -minDT 5