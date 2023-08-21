#!/bin/sh

## ogbg-molhiv
python sweep.py --model=Baseline --dataset=ogbg-molhiv
python sweep.py --model=GINE --dataset=ogbg-molhiv
python sweep.py --model=DropGINE --dataset=ogbg-molhiv
python sweep.py --model=PPGN --dataset=ogbg-molhiv
python sweep.py --model=SignNet --dataset=ogbg-molhiv
python sweep.py --model=ESAN --dataset=ogbg-molhiv
python sweep.py --model=GIN --dataset=ogbg-molhiv
python sweep.py --model=MeanAggrGINE --dataset=ogbg-molhiv
python sweep.py --model=GINEE --dataset=ogbg-molhiv


## ZINC12k
python sweep.py --model=Baseline --dataset=ZINC12k
python sweep.py --model=GINE --dataset=ZINC12k
python sweep.py --model=DropGINE --dataset=ZINC12k
python sweep.py --model=PPGN --dataset=ZINC12k
python sweep.py --model=SignNet --dataset=ZINC12k
python sweep.py --model=ESAN --dataset=ZINC12k
python sweep.py --model=GIN --dataset=ZINC12k
python sweep.py --model=MeanAggrGINE --dataset=ZINC12k
python sweep.py --model=GINEE --dataset=ZINC12k


## IMDB
python sweep.py --model=Baseline --dataset=IMDB-BINARY
python sweep.py --model=GINE --dataset=IMDB-BINARY
python sweep.py --model=DropGINE --dataset=IMDB-BINARY
python sweep.py --model=PPGN --dataset=IMDB-BINARY
python sweep.py --model=SignNet --dataset=IMDB-BINARY
python sweep.py --model=ESAN --dataset=IMDB-BINARY
python sweep.py --model=MeanAggrGINE --dataset=IMDB-BINARY
python sweep.py --model=GINEE --dataset=IMDB-BINARY

python sweep.py --model=Baseline --dataset=IMDB-MULTI
python sweep.py --model=GINE --dataset=IMDB-MULTI
python sweep.py --model=DropGINE --dataset=IMDB-MULTI
python sweep.py --model=PPGN --dataset=IMDB-MULTI
python sweep.py --model=SignNet --dataset=IMDB-MULTI
python sweep.py --model=ESAN --dataset=IMDB-MULTI
python sweep.py --model=MeanAggrGINE --dataset=IMDB-MULTI
python sweep.py --model=GINEE --dataset=IMDB-MULTI


## MUTAG
python sweep.py --model=GINE --dataset=MUTAG
python sweep.py --model=DropGINE --dataset=MUTAG
python sweep.py --model=PPGN --dataset=MUTAG
python sweep.py --model=SignNet --dataset=MUTAG
python sweep.py --model=ESAN --dataset=MUTAG
python sweep.py --model=MeanAggrGINE --dataset=MUTAG
python sweep.py --model=GIN --dataset=MUTAG
python sweep.py --model=Baseline --dataset=MUTAG
python sweep.py --model=GINEE --dataset=MUTAG


# MUTAG adversarial
python sweep.py --model=GINE --dataset=MUTAG --adversarial_training
python sweep.py --model=DropGINE --dataset=MUTAG --adversarial_training
python sweep.py --model=PPGN --dataset=MUTAG --adversarial_training
python sweep.py --model=SignNet --dataset=MUTAG --adversarial_training
python sweep.py --model=ESAN --dataset=MUTAG --adversarial_training
python sweep.py --model=MeanAggrGINE --dataset=MUTAG --adversarial_training
python sweep.py --model=GIN --dataset=MUTAG --adversarial_training
