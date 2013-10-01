#!/bin/bash

for test_file in resolver/*_test.py; do
  echo "Running test $test_file"
  PYTHONPATH=${PYTHONPATH}:. python $test_file
done
