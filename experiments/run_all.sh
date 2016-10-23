k=13
l=7
algo="bin_ovnmtf"
dataset="nips"
for ip in `cat ips_$algo.txt`;
do
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/biclustering/experiments/nips_data ubuntu@$ip:~/project/nips_data
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/biclustering/experiments/nips_labels ubuntu@$ip:~/project/nips_labels
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/biclustering/experiments/run_algo.py ubuntu@$ip:~/project/run_algo.py
    scp -i ~/.ssh/lucasbrunialti2.pem ~/git/nmtf-coclustering/algos.cpp ubuntu@$ip:~/project/algos.cpp

    ssh -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip "
        cd ~/project/ &&
        algo=$algo &&
        k=$k &&
        g++ -std=c++11 algos.cpp -o algos -lopenblas -O3 -march=native -larmadillo -lopenblas -llapack -fopenmp -lhdf5 &&
        nohup python run_algo.py -d $dataset -a $algo -k $k -l $l > ${algo}_k=${k}_l=${l}_${dataset}_log.txt 2>&1 &
    " &

    l=$(($l+3))
done;
