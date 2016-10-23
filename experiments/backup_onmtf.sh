k=13
l=10
algo="onmtf"
dataset="ig"
for ip in `cat /Users/lucasbrunialti/git/biclustering/experiments/ips_onmtf.txt`;
do
    mkdir -p /Users/lucasbrunialti/git/biclustering/experiments/backup/${algo}/
    scp -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip:~/${dataset}_kk=${k}_ll=${l}_${algo}_news_results.csv backup/${algo}/

    l=$(($l+3))
done;
