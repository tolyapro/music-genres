#!/bin/bash
for i in {0..799}
do
   sox "$i.au" "$i.wav" 
done
