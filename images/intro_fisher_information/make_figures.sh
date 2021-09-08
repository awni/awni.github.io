#!/bin/bash

python plots.py

latexmk -pdf log_prob.tikz
latexmk -c log_prob.tikz
pdf2svg log_prob.pdf log_prob.svg
rm log_prob.pdf
