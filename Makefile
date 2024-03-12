fit:
	python ./src/Trainer/app.py
to_js:
	tensorflowjs_converter --input_format keras ModelNumbers.h5 tfjs
	ls
view_struct:
	python ./src/Trainer/summary.py
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
run_server:
	python -m http.server 8000
test_01:
	python ./src/Tests/test_01.py
all:
	python ./src/Trainer/app.py
	python ./src/Tests/test_01.py
	tensorflowjs_converter --input_format keras ModelNumbers.h5 tfjs
	python -m http.server 8000