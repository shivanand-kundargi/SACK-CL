#!/bin/bash


for seed in 0 1 2
do
        echo "Starting iCaRL run with seed $seed"
        python main.py \
        --dataset=seq-cub200 \
        --model=icarl \
        --buffer_size=2000 \
        --model_config=best \
        --cog_cl 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-icarl-cub200-SACK-mammoth \
        --wandb_name=SACK-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed  

        python main.py \
        --dataset=seq-cub200 \
        --model=der \
        --backbone=resnet50 \
        --buffer_size=500 \
        --model_config=best \
        --cog_cl 1 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-der-cub200-SACK-mammoth \
        --wandb_name=SACK-$seed \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed 


        python main.py \
                --dataset=seq-cub200 \
                --model=derpp \
                --buffer_size=2000 \
                --backbone=resnet50 \
                --model_config=best \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-derpp-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --seed=$seed 


        python main.py \
                --dataset=seq-cub200 \
                --model=lwf \
                --lr=0.03\
                --backbone=resnet50 \
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-lwf-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --seed=$seed


        python main.py \
                --dataset=seq-cub200 \
                --model=coda_prompt \
                --model_config=best\
                --cog_cl 1 \
                --sack_scores_type=0 \
                --wandb_entity=abcxyz8431-cl \
                --wandb_project=Final-coda_prompt-cub200-SACK-mammoth \
                --wandb_name=SACK-$seed \
                --enable_other_metrics=True \
                --savecheck=task \
                --permute_classes=True \
                --seed=$seed

done













# python main.py \
#         --dataset=seq-cub200 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-uniform2weights \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed  

# python main.py \
#     --dataset=seq-cub200 \
#     --model=der \
#     --backbone=resnet50 \
#     --buffer_size=500 \
#     --model_config=best \
#     --cog_cl 1 \
#     --sack_scores_type=1 \
#     --wandb_entity=abcxyz8431-cl \
#     --wandb_project=Final-der-cub200-cogcl-mammoth \
#     --wandb_name=cogcl-run-seed-uniform2weights \
#     --enable_other_metrics=True \
#     --savecheck=task \
#     --permute_classes=True \
#     --seed=$seed 


# python main.py \
#         --dataset=seq-cub200 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --backbone=resnet50 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-uniform2weights \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed 

# python main.py \
#         --dataset=seq-cub200 \
#         --model=lwf \
#         --lr=0.03\
#         --backbone=resnet50 \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-uniform2weights \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed

python main.py \
        --dataset=seq-cub200 \
        --model=coda_prompt \
        --model_config=best\
        --cog_cl 1 \
        --sack_scores_type=1 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cub200-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-uniform2weights \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed
























# python main.py \
#         --dataset=seq-cub200 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-random244 \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed  

# python main.py \
#     --dataset=seq-cub200 \
#     --model=der \
#     --backbone=resnet50 \
#     --buffer_size=500 \
#     --model_config=best \
#     --cog_cl 1 \
#     --sack_scores_type=2 \
#     --wandb_entity=abcxyz8431-cl \
#     --wandb_project=Final-der-cub200-cogcl-mammoth \
#     --wandb_name=cogcl-run-seed-random244 \
#     --enable_other_metrics=True \
#     --savecheck=task \
#     --permute_classes=True \
#     --seed=$seed 


# python main.py \
#         --dataset=seq-cub200 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --backbone=resnet50 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-random244 \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed 

# python main.py \
#         --dataset=seq-cub200 \
#         --model=lwf \
#         --lr=0.03\
#         --backbone=resnet50 \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-random244 \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed

python main.py \
        --dataset=seq-cub200 \
        --model=coda_prompt \
        --model_config=best\
        --cog_cl 1 \
        --sack_scores_type=2 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cub200-cogcl-mammoth \
        --wandb_name=cogcl-run-seed-random244 \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed

















# python main.py \
#         --dataset=seq-cub200 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cub200-cogcl-mammoth \
#         --wandb_name=original-run \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True 



# python main.py \
#         --dataset=seq-cub200 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cub200-cogcl-mammoth \
#         --wandb_name=original-run \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed  

# python main.py \
#     --dataset=seq-cub200 \
#     --model=der \
#     --backbone=resnet50 \
#     --buffer_size=500 \
#     --model_config=best \
#     --cog_cl 0 \
#     --sack_scores_type=0 \
#     --wandb_entity=abcxyz8431-cl \
#     --wandb_project=Final-der-cub200-cogcl-mammoth \
#     --wandb_name=original-run  \
#     --enable_other_metrics=True \
#     --savecheck=task \
#     --permute_classes=True \
#     --seed=$seed 


# python main.py \
#         --dataset=seq-cub200 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --backbone=resnet50 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cub200-cogcl-mammoth \
#         --wandb_name=original-run  \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed 

# python main.py \
#         --dataset=seq-cub200 \
#         --model=lwf \
#         --lr=0.003\
#         --backbone=resnet50 \
#         --cog_cl 0 \
#         --sack_scores_type=0 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cub200-cogcl-mammoth \
#         --wandb_name=original-run  \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed

python main.py \
        --dataset=seq-cub200 \
        --model=coda_prompt \
        --model_config=best\
        --cog_cl 0 \
        --sack_scores_type=0 \
        --wandb_entity=abcxyz8431-cl \
        --wandb_project=Final-coda_prompt-cub200-cogcl-mammoth \
        --wandb_name=original-run  \
        --enable_other_metrics=True \
        --savecheck=task \
        --permute_classes=True \
        --seed=$seed


# for seed in 0 1 2
# do
#     echo "Starting iCaRL run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-icarl-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=icarl-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \


#     echo "completed iCaRL run with seed $seed"    
# done






# for seed in 0 1 2
# do
#     echo "Starting GEM run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=gem \
#         --buffer_size=2000 \
#         --lr=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-gem-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=gem-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dpp" \
#     echo "completed GEM run with seed $seed"

# done






# for seed in 0 1 2
# do
#     echo "Starting ER run with $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=er \
#         --buffer_size=2000 \
#         --lr=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-er-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=er-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      
#     echo "completed ER run with seed $seed"    
# done







# for seed in 0 1 2
# do
#     echo "Starting der run with $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=der \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-der-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=der-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       
#     echo "completed der run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting derpp run with seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-derpp-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=derpp-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed derpp run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting bic run with seed  $seed"    
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=bic \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-bic-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=bic-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      
#     echo "completed bic run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting gss run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=gss \
#         --buffer_size=2000 \
#         --n_epochs=50 \
#         --lr=0.05 \
#         --gss_minibatch_size=10 \
#         --batch_size=10 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-gss-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=gss-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      

#     echo "completed gss run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting hal run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=hal \
#         --buffer_size=2000 \
#         --hal_lambda=0.1 \
#         --lr=0.03 \
#         --beta=0.3 \
#         --gamma=0.1 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-hal-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=hal-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \


#     echo "completed hal run with seed $seed "    
# done







# for seed in 0 1 2
# do

#     echo "Starting mer run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=mer \
#         --buffer_size=2000 \
#         --minibatch_size 25 \
#         --n_epochs 50 \
#         --lr=0.1 \
#         --beta=0.01 \
#         --gamma=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-mer-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=mer-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \ 
#     echo "completed mer run with seed $seed "          
# done






# for seed in 0 1 2
# do
#     echo "Starting lwf run with seed $seed "
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=lwf \
#         --lr=0.0.\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-lwf-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=lwf-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       
#     echo "completed lwf run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting coda-prompt run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=coda_prompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-coda_prompt-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=codaprompt-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed coda-prompt run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting dualprompt run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=dualprompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-dualprompt-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=daulprompt-cub200-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed dualprompt run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting dap run with seed $seed"
#     python main.py \
#         --dataset=seq-cub200 \
#         --model=dap \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-dap-cub200-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=dap-cub200-cogcl-run-seed-$seed  \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \        

#     echo "completed dap run with seed $seed"    
# done