fit:
	py ./src/Trainer/app.py
clean_datasets:
	rm ./Datasets/ -r
clean_model:
	rm ./ModelNumbers.h5
clean_modeljs:
	rm ./tfjs_model -r -f
clean_all:
	rm ./Datasets/ -r
	rm ./ModelNumbers.h5
	rm ./tfjs_model -r -f
test_01:
	py ./src/Tests/test_01.py