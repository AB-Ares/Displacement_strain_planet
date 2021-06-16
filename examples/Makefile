#   Convert notebooks to html.
#   Cleanup output data

JUPYTER ?= jupyter nbconvert
PYTHON = python3

SHELL=/bin/sh

ODIR = ../docs/_build/html/notebooks
JFLAGS = --to html --execute --ExecutePreprocessor.timeout=120 --output-dir=$(ODIR)

IPYNB = Run_demo.ipynb

HTML = $(addprefix $(ODIR)/, Run_demo.html)

.PHONY: all run-notebooks clean

all: $(HTML)

clean:
	@-rm -f data/*.npy

$(ODIR)/%.html: %.ipynb
	$(JUPYTER) $(JFLAGS) $<