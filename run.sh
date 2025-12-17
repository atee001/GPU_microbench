#!/bin/bash

make clean
make
./bin/benchmark > record.log
python3 plot.py
