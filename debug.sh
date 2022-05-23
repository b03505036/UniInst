#!/bin/bash
 
sum=0
i=1
 
while(( i <= 100 ))
do
  let "sum=i"
  sleep 10
  echo "$sum"
done
