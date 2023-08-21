#!/bin/sh

i=0
# for i in {0..9}
# do
#     python train.py ...
# done


## MolHIV
python train.py --model=GINE --dataset=ogbg-molhiv --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=64
python train.py --model=DropGINE --dataset=ogbg-molhiv --hidden_units=150 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=PPGN --dataset=ogbg-molhiv --hidden_units=64 --num_layers=5 --dropout=0 --lr=0.0001 --seed=$i --batch_size=16
python train.py --model=SignNet --dataset=ogbg-molhiv --hidden_units=95 --num_layers=16 --pos_enc_dim=37 --dropout=0 --lr=0.001 --seed=$i --patience=200 --epochs=1000 --batch_size=64 
python train.py --model=ESAN --dataset=ogbg-molhiv --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0.5 --lr=0.01 --seed=$i --batch_size=64

python train.py --model=Baseline --dataset=ogbg-molhiv --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=GIN --dataset=ogbg-molhiv --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=MeanAggrGINE --dataset=ogbg-molhiv --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32


## ZINC12k
python train.py --model=GINE --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=DropGINE --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=PPGN --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.0001 --seed=$i --batch_size=32
python train.py --model=SignNet --dataset=ZINC12k --hidden_units=95 --num_layers=16 --pos_enc_dim=37 --dropout=0 --lr=0.001 --seed=$i --patience=400 --epochs=1000 --batch_size=128 
python train.py --model=ESAN --dataset=ZINC12k --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0.5 --lr=0.001 --seed=$i --batch_size=64

python train.py --model=Baseline --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=GIN --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=MeanAggrGINE --dataset=ZINC12k --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32


## IMDB-BINARY
python train.py --model=Baseline --dataset=IMDB-BINARY --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=GINE --dataset=IMDB-BINARY --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=DropGINE --dataset=IMDB-BINARY --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=64
python train.py --model=PPGN --dataset=IMDB-BINARY --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.001 --seed=$i --batch_size=16
python train.py --model=SignNet --dataset=IMDB-BINARY --hidden_units=95 --num_layers=16 --pos_enc_dim=30 --dropout=0 --lr=0.01 --seed=$i --patience=200 --epochs=1000 --batch_size=128
python train.py --model=ESAN --dataset=IMDB-BINARY --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0.5 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=MeanAggrGINE --dataset=IMDB-BINARY --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32


## IMDB-MULTI
python train.py --model=Baseline --dataset=IMDB-MULTI --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=GINE --dataset=IMDB-MULTI --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=DropGINE --dataset=IMDB-MULTI --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=PPGN --dataset=IMDB-MULTI --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.0001 --seed=$i --batch_size=32
python train.py --model=SignNet --dataset=IMDB-MULTI --hidden_units=95 --num_layers=16 --pos_enc_dim=15 --dropout=0 --lr=0.01 --seed=$i --patience=200 --epochs=1000 --batch_size=128
python train.py --model=ESAN --dataset=IMDB-MULTI --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0 --lr=0.01 --seed=$i --batch_size=128
python train.py --model=MeanAggrGINE --dataset=IMDB-MULTI --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64


## MUTAG
python train.py --model=GINE --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.01 --seed=$i --batch_size=32
python train.py --model=DropGINE --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=32
python train.py --model=PPGN --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0.5 --lr=0.0001 --seed=$i --batch_size=64
python train.py --model=SignNet --dataset=MUTAG --hidden_units=95 --num_layers=16 --pos_enc_dim=30 --dropout=0 --lr=0.01 --seed=$i --patience=200 --epochs=1000 --batch_size=128
python train.py --model=ESAN --dataset=MUTAG --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0 --lr=0.001 --seed=$i --batch_size=64

python train.py --model=MeanAggrGINE --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=64
python train.py --model=GIN --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=32
python train.py --model=Baseline --dataset=MUTAG --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=32

## MUTAG adversarial
python train.py --model=GINE --dataset=MUTAG --adversarial_training --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=64
python train.py --model=DropGINE --dataset=MUTAG --adversarial_training --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=64
python train.py --model=PPGN --dataset=MUTAG --adversarial_training --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=SignNet --dataset=MUTAG --adversarial_training --hidden_units=95 --num_layers=16 --pos_enc_dim=30 --dropout=0 --lr=0.001 --seed=$i --patience=200 --epochs=1000 --batch_size=128
python train.py --model=ESAN --dataset=MUTAG --adversarial_training --hidden_units=64 --num_layers=5 --sample_fraction=0.2 --policy=ego_nets_plus --num_hops=3 --dropout=0 --lr=0.001 --seed=$i --batch_size=64

python train.py --model=MeanAggrGINE --dataset=MUTAG --adversarial_training --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.001 --seed=$i --batch_size=32
python train.py --model=GIN --dataset=MUTAG --adversarial_training --hidden_units=300 --num_layers=5 --dropout=0 --lr=0.01 --seed=$i --batch_size=32

