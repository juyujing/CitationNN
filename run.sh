## CitationNN

## CiteULike
python -u main.py --seed 2021 --dataset citeulike --att_dropout 1 --step 5 --lr 0.001 --l2 1e-6  --pool sum --load_ii_sort 50 --context_hops 1 --with_sim 1 --with_user 1 --e 0.001 --with_uu_co_author 0 --with_ii_co_author 0 --with_uu_co_venue 0 --with_ii_co_venue 0 --gpu_id 0 --gnn MCAP --batch_size 1024 --fast_test 1
python -u main.py --seed 2021 --dataset citeulike --att_dropout 1 --step 5 --lr 0.001 --l2 1e-6  --pool sum --load_ii_sort 50 --context_hops 1 --with_sim 1 --with_user 1 --e 0.001 --with_uu_co_author 0 --with_ii_co_author 0 --with_uu_co_venue 0 --with_ii_co_venue 0 --gpu_id 0 --gnn CitationNN --batch_size 1024 --alpha 0.8 --fast_test 1

## dblp
python -u main.py --seed 2021 --context_hops 1 --dataset dblp --lr 0.001 --l2 0.0001  --att_dropout 0 --pool sum --with_sim 0 --with_user 0 --gpu_id 0 --gnn MCAP --batch_size 1024 --step 1
python -u main.py --seed 2021 --context_hops 1 --dataset dblp --lr 0.001 --l2 1e-6  --att_dropout 1 --pool sum --with_sim 0 --with_user 0 --gpu_id 0 --gnn CitationNN --batch_size 1024 --alpha 0.95
