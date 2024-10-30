export PROJECT_DIR=/users/mlepori/data/mlepori/projects/relational-circuits/

controls=("none")
models=("clip")
analysis=("shape")
compositional=(-1)

ids_256=("n1wy4pyq")

lr=(0.1 0.01 0.001 0.0001)
mask_lr=(0.1 0.01 0.001 0.0001)

for model_idx in {0..0};
do
    for analysis_idx in {0..0};
    do
        for lr_idx in {0..3};
        do
            for mask_idx in {0..3};
            do
                for control_idx in {0..0};
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
                    export LR=${lr[$lr_idx]}
                    export MASK_LR=${mask_lr[$mask_idx]}

                    export JOBNAME=${MODEL}_${ANALYSIS}_${RUN_ID}_${LR}_${MASK_LR}
                    echo "$JOBNAME"
                    sbatch -J $JOBNAME -o CCV/out/${JOBNAME}.out -e CCV/err/${JOBNAME}.err $PROJECT_DIR/CCV/disc_das.script
                    done
                done
            done
        done
    done
done


