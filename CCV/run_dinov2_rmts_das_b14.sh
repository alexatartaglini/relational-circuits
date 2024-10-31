export PROJECT_DIR=/users/mlepori/data/mlepori/projects/relational-circuits/

controls=("none" "wrong_object") 
models=("dinov2_vit")
analysis=("shape" "color")
compositional=(-1 32)

ids_32=("udjgzn8d")
ids_256=("okppu3qm")


for model_idx in {0..0};
do
    for analysis_idx in {0..1};
    do
        for control_idx in {0..1};
        do
            for compositional_idx in {0..1};
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
                export PATCH_SIZE=14
                export OBJ_SIZE=28
                export LR=0.001
                export MASK_LR=0.01
                export JOBNAME=${MODEL}_${ANALYSIS}_${CONTROL}_${COMPOSITIONAL}_${RUN_ID}
                echo "$JOBNAME"
                sbatch -J $JOBNAME -o CCV/out/${JOBNAME}.out -e CCV/err/${JOBNAME}.err $PROJECT_DIR/CCV/rmts_das.script
            done
        done
    done
done


