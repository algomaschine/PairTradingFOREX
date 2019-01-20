cd "C:\Users\edward\Documents\Pair Trading\FX\March\"
python application.py -input EURAUDpro60.csv -output nEURAUDpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87
python application.py -input EURJPYpro60.csv -output nEURJPYpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input EURNOKpro60.csv -output nEURNOKpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input EURNZDpro60.csv -output nEURNZDpro60pro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input GBPAUDpro60.csv -output nGBPAUDpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input GBPNZDpro60.csv -output nGBPNZDpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input USDCADpro60.csv -output nUSDCADpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input USDCHFpro60.csv -output nUSDCHFpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input USDJPYpro60.csv -output nUSDJPYpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

python application.py -input USDNOKpro60.csv -output nUSDNOKpro60.csv -type quantile -q_lower 0.13 -q_upper 0.87

@pause
