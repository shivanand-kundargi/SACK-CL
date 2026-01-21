#!/bin/bash


#SACK(W->U)
# python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=icarl \
#                 --buffer_size=2000 \
#                 --model_config=best \
#                 --cog_cl 0 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-icarl-cifar100-SACK-mammoth \
#                 --wandb_name=original-run-seed-0\
#                 --enable_other_metrics=True \
#                 --log_perf_metrics=1 \
#                 --permute_classes=True \
#                 --seed=0


# python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=icarl \
#                 --buffer_size=2000 \
#                 --model_config=best \
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-icarl-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-seed-0\
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --log_perf_metrics=1 \
#                 --seed=0




# for seed in 0 1 2
# do
#         echo "Starting iCaRL run with seed $seed"   
#         python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=icarl \
#                 --buffer_size=2000 \
#                 --model_config=best \
#                 --n_epochs=5 \
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-icarl-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-$seed\
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --seed=$seed  


#         python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=der \
#                 --buffer_size=2000 \
#                 --model_config=best \
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-der-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-$seed \
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --seed=$seed

#         python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=derpp \
#                 --buffer_size=2000 \
#                 --model_config=best \
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-derpp-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-$seed \
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --seed=$seed

#         python main.py \
#                 --dataset=seq-cifar100 \
#                 --model=lwf \
#                 --lr=0.003\
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-lwf-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-$seed \
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --seed=$seed

#         python main.py \
#                 --dataset=seq-cifar100-224 \
#                 --model=coda_prompt \
#                 --model_config=best\
#                 --cog_cl 1 \
#                 --sack_scores_type=0 \
#                 --wandb_entity=abcxyz8431-cl \
#                 --wandb_project=Final-coda_prompt-cifar100-SACK-mammoth \
#                 --wandb_name=SACK-$seed \
#                 --enable_other_metrics=True \
#                 --permute_classes=True \
#                 --seed=$seed
# done












# #SACK(U->W)


# # python main.py \
# #         --dataset=seq-cifar100 \
# #         --model=icarl \
# #         --buffer_size=2000 \
# #         --model_config=best \
# #         --cog_cl 1 \
# #         --sack_scores_type=1 \
# #         --wandb_entity=abcxyz8431-cl \
# #         --wandb_project=Final-icarl-cifar100-cogcl-mammoth \
# #         --wandb_name=cogcl-run-seed-$seed-Uniform2Weights\
# #         --enable_other_metrics=True \
# #         --permute_classes=True 


# # python main.py \
# #         --dataset=seq-cifar100 \
#         --model=der \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-der-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-Uniform2Weights \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-Uniform2Weights \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=lwf \
#         --lr=0.03\
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-Uniform2Weights \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100-224 \
#         --model=coda_prompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --sack_scores_type=1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-Uniform2Weights \
#         --enable_other_metrics=True \
#         --permute_classes=True





#Random scores


# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224\
#         --enable_other_metrics=True \
#         --permute_classes=True 


# python main.py \
#         --dataset=seq-cifar100 \
#         --model=der \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-der-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 
                                        
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=lwf \
#         --lr=0.03\
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100-224 \
#         --model=coda_prompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True
























#original run with no SACK scores
# python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cifar100-original-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224\
#         --enable_other_metrics=True \
#         --permute_classes=True 


# python main.py \
#         --dataset=seq-cifar100 \
#         --model=der \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-der-cifar100-original-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 0 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-derpp-cifar100-original-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=lwf \
#         --lr=0.03\
#         --cog_cl 0 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-lwf-cifar100-original-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True 

# python main.py \
#         --dataset=seq-cifar100-224 \
#         --model=coda_prompt \
#         --model_config=best\
#         --cog_cl 0 \
#         --sack_scores_type=2 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-coda_prompt-cifar100-original-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores224 \
#         --enable_other_metrics=True \
#         --permute_classes=True

# python main.py \
#         --dataset=seq-cifar100 \
#         --model=bic \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-bic-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores\
#         --enable_other_metrics=True \
#         --permute_classes=True


# for seed in 0 1 2
# do
#     echo "Starting iCaRL run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=icarl \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=abcxyz8431-cl \
#         --wandb_project=Final-icarl-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores\
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=icarl-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \


#     echo "completed iCaRL run with seed $seed"    
# done






# for seed in 0 1 2
# do
#     echo "Starting GEM run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=gem \
#         --buffer_size=2000 \
#         --lr=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-gem-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=gem-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dpp" \
#     echo "completed GEM run with seed $seed"

# done






# for seed in 0 1 2
# do
#     echo "Starting ER run with $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=er \
#         --buffer_size=2000 \
#         --lr=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-er-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed-randomscores \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=er-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      
#     echo "completed ER run with seed $seed"    
# done







# for seed in 0 1 2
# do
#     echo "Starting der run with $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=der \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-der-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=der-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       
#     echo "completed der run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting derpp run with seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=derpp \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-derpp-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=derpp-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed derpp run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting bic run with seed  $seed"    
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=bic \
#         --buffer_size=2000 \
#         --model_config=best \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-bic-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=bic-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      
#     echo "completed bic run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting gss run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=gss \
#         --buffer_size=2000 \
#         --n_epochs=50 \
#         --lr=0.05 \
#         --gss_minibatch_size=10 \
#         --batch_size=10 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-gss-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=gss-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \      

#     echo "completed gss run with seed $seed"    
# done





# for seed in 0 1 2
# do
#     echo "Starting hal run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=hal \
#         --buffer_size=2000 \
#         --hal_lambda=0.1 \
#         --lr=0.03 \
#         --beta=0.3 \
#         --gamma=0.1 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-hal-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=hal-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \


#     echo "completed hal run with seed $seed "    
# done







# for seed in 0 1 2
# do

#     echo "Starting mer run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=mer \
#         --buffer_size=2000 \
#         --minibatch_size 25 \
#         --n_epochs 50 \
#         --lr=0.1 \
#         --beta=0.01 \
#         --gamma=0.03 \
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-mer-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=mer-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \ 
#     echo "completed mer run with seed $seed "          
# done






# for seed in 0 1 2
# do
#     echo "Starting lwf run with seed $seed "
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=lwf \
#         --lr=0.03\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-lwf-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=lwf-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       
#     echo "completed lwf run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting coda-prompt run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=coda_prompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-coda_prompt-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=codaprompt-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed coda-prompt run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting dualprompt run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=dualprompt \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-dualprompt-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=daulprompt-cifar100-cogcl-run-seed-$seed \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \       

#     echo "completed dualprompt run with seed $seed"    
# done






# for seed in 0 1 2
# do

#     echo "Starting dap run with seed $seed"
#     python main.py \
#         --dataset=seq-cifar100 \
#         --model=dap \
#         --model_config=best\
#         --cog_cl 1 \
#         --wandb_entity=shiva-umbc \
#         --wandb_project=Final-dap-cifar100-cogcl-mammoth \
#         --wandb_name=cogcl-run-seed-$seed \
#         --enable_other_metrics=True \
#         --savecheck=task \
#         --permute_classes=True \
#         --seed=$seed \
#         --ckpt_name=dap-cifar100-cogcl-run-seed-$seed  \
#         # --device='0,1,2,3' \
#         # --distributed="dp" \        

#     echo "completed dap run with seed $seed"    
# done