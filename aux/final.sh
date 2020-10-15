#!/bin/bash

# for i in $(seq 1 30);
# do
#   ./ACO_red bases_grafos/entrada1.txt 150 200 0.3 1 3 FINAL-entrada1-"$i"
# done

# for i in $(seq 1 30);
# do
#   ./ACO_red bases_grafos/entrada2.txt 150 2000 0.3 1 3 FINAL-entrada2-"$i"
# done

for i in $(seq 1 30);
do
  ./ACO_red bases_grafos/entrada3.txt 200 2000 0.1 1 3 FINAL-entrada3-"$i"
done

