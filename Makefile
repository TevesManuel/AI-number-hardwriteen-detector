fit:
	python ./src/Trainer/app.py
to_js:
	pip install tensorflowjs
	mkdir tfjs_target_dir
	tensorflowjs_converter --input_format keras ModelNumbers.h5 tfjs_target_dir
	ls
view_struct:
	py ./src/Trainer/summary.py
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
	python ./src/Tests/test_01.py