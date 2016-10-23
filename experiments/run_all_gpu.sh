l=7
algo="ovnmtf"
for ip in `cat ips_$algo.txt`;
do
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/biclustering/experiments/run_algo.py ubuntu@$ip:~/run_algo.py
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/nmtf-coclustering/algos_gpu.cpp ubuntu@$ip:~/algos_gpu.cpp
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/biclustering/experiments/all_news_df.pkl ubuntu@$ip:~/all_news_df.pkl

    ssh -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip "
        nvcc --x=cu --std=c++11 algos_gpu.cpp -o algos -O3 -arch=sm_20 -larmadillo -lcublas -lcudart -lhdf5 -lopenblas &&
        algo=$algo &&
        l=$l &&
        nohup python run_algo.py -d ig -a $algo -k 13 -l $l > ${algo}_k=13_l=${l}_ig_log.txt 2>&1 &
    " &

    l=$(($l+3))
done;
