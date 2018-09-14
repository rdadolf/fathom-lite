#!/bin/bash

TMPDIR=`pwd`
#TMPDIR=/tmp/dataset

if [ ! -d $TMPDIR ]; then mkdir $TMPDIR; fi

cd $TMPDIR

curl -O http://www.statmt.org/wmt15/dev-v2.tgz
curl -O http://www.statmt.org/wmt10/training-giga-fren.tar

tar xzf dev-v2.tgz
tar xf training-giga-fren.tar
rm training-giga-fren.tar
gunzip giga-fren.release2.fixed.en.gz
gunzip giga-fren.release2.fixed.fr.gz

echo "Files saved in $TMPDIR"
