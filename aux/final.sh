#!/bin/bash

for i in $(seq 1 30);
do
  ./ACO_red bases_grafos/entrada1.txt 200 300 0.3 1 3 entrada1-"$i"-final
done

for i in $(seq 1 30);
do
  ./ACO_red bases_grafos/entrada2.txt 200 300 0.3 1 3 entrada2-"$i"-final
done

for i in $(seq 1 30);
do
  ./ACO_red bases_grafos/entrada3.txt 200 300 0.3 1 3 entrada3-"$i"-final
done

