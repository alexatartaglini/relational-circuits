export PROJECT_DIR=/users/mlepori/data/mlepori/projects/relational-circuits/

controls=("none" "random_patch" "wrong_object")
models=("clip" "imagenet" "dino" "scratch")
analysis=("shape" "color")
compositional=(-1 32)

ids_256=("n1wy4pyq" "mjfmy0s2" "j9omwocg" "fmr73rre")
ids_32=("q3zevmig" "cpzhfnou" "qosnu09a" "t3ankf4l")


for model_idx in {0..3};
do
    for analysis_idx in {0..1};
    do
        for control_idx in {0..2};
        do
            for compositional_idx in {0..0};
            do
                export MODEL=${models[$model_idx]}
                export ANALYSIS=${analysis[$analysis_idx]}
                export CONTROL=${controls[$control_idx]}
                export COMPOSITIONAL=${compositional[$compositional_idx]}
                if [ $compositional_idx -eq 0 ]
                then
                export RUN_ID=${ids_256[$model_idx]}
                else
                export RUN_ID=${ids_32[$model_idx]}
                fi
                export JOBNAME=${MODEL}_${ANALYSIS}_${CONTROL}_${COMPOSITIONAL}_${RUN_ID}
                echo "$JOBNAME"
                sbatch -J $JOBNAME -o CCV/out/${JOBNAME}.out -e CCV/err/${JOBNAME}.err $PROJECT_DIR/CCV/run.script
            done
        done
    done
done


