#! /bin/bash
source ~/tf2/bin/activate 
python struc2vec/src/main.py --input $1 --output $2 --dimensions $3
rm -f random_walks.txt
rm -f struc2vec.log

