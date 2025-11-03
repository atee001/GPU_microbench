#!/bin/bash

make 
./bin/benchmark > record.log
python3 plot.py
