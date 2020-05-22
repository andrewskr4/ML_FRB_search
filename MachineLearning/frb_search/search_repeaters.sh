dm=349.7
for i in /data/repeaters/R3/*; do
    if [ -d "$i" ]; then
        cd $i
        mjd=$(basename "$i")
        gen_samples *.fil -d dm /data/andrew/flags_list.dat
        python /home/andrew/ML_FRB_search/MachineLearning/frb_search/run_model.py $i R3 mjd
    fi
done    
