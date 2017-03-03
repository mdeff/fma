NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

install:
	pip install --upgrade pip
	pip install --upgrade numpy
	pip install --upgrade -r requirements.txt

readme:
	grip README.md

.PHONY: run $(NB) clean install readme
