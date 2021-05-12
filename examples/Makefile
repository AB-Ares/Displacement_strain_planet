#   Convert notebooks to html.

JUPYTER ?= jupyter nbconvert
PYTHON = python3

SHELL=/bin/sh

ODIR = ../docs/notebooks
JFLAGS = --to html --execute --ExecutePreprocessor.timeout=120 --output-dir=$(ODIR)

IPYNB = Run_demo.ipynb

HTML = $(addprefix $(ODIR)/, Run_demo.html)

.PHONY: clean all run-notebooks

all: $(HTML)

$(ODIR)/%.html: %.ipynb
	$(JUPYTER) $(JFLAGS) $<