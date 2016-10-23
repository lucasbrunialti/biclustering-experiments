k=3
l=2
algo="onmtf"
dataset="igtoy"
for ip in `cat ips_$algo.txt`;
do
    mkdir -p backup/${algo}/
    scp -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip:~/project/${algo}_k=${k}_l=${l}_${dataset}_log.txt backup/${algo}/
    scp -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip:~/project/${dataset}_kk=${k}_ll=${l}_${algo}_news_results.csv backup/${algo}/

    l=$(($l+1))
done;
