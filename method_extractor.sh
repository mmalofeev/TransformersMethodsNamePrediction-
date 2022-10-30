#!/bin/zsh

cd parser
./gradlew run --args="$1 $2"
