#!/bin/bash
TMPDIR=`pwd`
#TMPDIR=/tmp/dataset

if [ ! -d $TMPDIR ]; then mkdir $TMPDIR; fi

cd $TMPDIR

curl -O http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz

tar xzf tasks_1-20_v1-2.tar.gz

echo "Files saved in $TMPDIR"
