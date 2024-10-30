export PROJECT_DIR=/users/mlepori/data/mlepori/projects/relational-circuits/

controls=("false" "true") 
#models=("clip")
models=("dinov2_vit")

#patch_size=(16 32)
patch_size=(14)

compositional=(-1 32)

#ids_32=("ld8hk72f" "91zt5xij")
#ids_256=("tavk2a8o" "3eymil2v")

ids_32=("udjgzn8d")
ids_256=("okppu3qm")
for model_idx in {0..0};
do
    for patch_idx in {0..0};
    do
        for control_idx in {0..1};
        do
            for compositional_idx in {0..1};
            do
                export MODEL=${models[$model_idx]}
                export PATCH_SIZE=${patch_size[$patch_idx]}
                export CONTROL=${controls[$control_idx]}
                export COMPOSITIONAL=${compositional[$compositional_idx]}
                if [ $compositional_idx -eq 0 ]
                then
                export RUN_ID=${ids_256[$patch_idx]}
                else
                export RUN_ID=${ids_32[$patch_idx]}
                fi
                export ALPHA=1.0
                export OBJ_SIZE=28
                export JOBNAME=${MODEL}_${PATCH_SIZE}_${CONTROL}_${COMPOSITIONAL}_${RUN_ID}_${ALPHA}
                echo "$JOBNAME"
                sbatch -J $JOBNAME -o CCV/out/${JOBNAME}.out -e CCV/err/${JOBNAME}.err $PROJECT_DIR/CCV/linear_intervention.script
            done
        done
    done
done


