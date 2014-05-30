#!/bin/bash
for i in {0..199}
do
   sox "$i.au" "$i.wav" 
done
