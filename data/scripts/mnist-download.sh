#!/bin/bash

TMPDIR=`pwd`
#TMPDIR=/tmp/dataset

if [ ! -d $TMPDIR ]; then mkdir $TMPDIR; fi

cd $TMPDIR

curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "Files saved in $TMPDIR"
