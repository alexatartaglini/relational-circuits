export PROJECT_DIR=/users/mlepori/data/mlepori/projects/relational-circuits/
export CONFIG_DIR=configs/Tracing/imagenet/
for file in ${CONFIG_DIR}/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/relational-circuits/${CONFIG_DIR}/${JOBNAME}.yaml


    sbatch -J $JOBNAME -o CCV/out/${JOBNAME}.out -e CCV/err/${JOBNAME}.err $PROJECT_DIR/CCV/run.script
done

