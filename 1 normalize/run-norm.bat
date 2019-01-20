cd "C:\Users\edward\Documents\Pair Trading\FX\normalize"
python application.py -input AUDJPYpro240.csv -output nA.csv -type quantile -q_lower 0.13 -q_upper 0.87
python application.py -input AUDNZDpro240.csv -output nB.csv -type quantile -q_lower 0.13 -q_upper 0.87
@pause
