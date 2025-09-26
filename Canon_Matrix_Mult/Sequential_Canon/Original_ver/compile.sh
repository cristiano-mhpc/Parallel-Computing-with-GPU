#!/bin/bash

mpic++ main.cpp -o cart.x -lopenblas -O3 -Wall -march=native
