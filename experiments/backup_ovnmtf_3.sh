k=13
l=7
algo="ovnmtf"
dataset="ig"
for ip in `cat /Users/lucasbrunialti/git/biclustering/experiments/ips_ovnmtf_3.txt`;
do
    mkdir -p /Users/lucasbrunialti/git/biclustering/experiments/backup/${algo}/
    scp -i ~/.ssh/lucasbrunialti2.pem ubuntu@$ip:~/${dataset}_kk=${k}_ll=${l}_X=X_train_tfidf_${algo}_news_results.csv backup/${algo}/

    l=$(($l+3))
done;
