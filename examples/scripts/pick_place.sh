data=all_pickplace_except_tk6
tdatas=('toykitchen6_croissant_out_of_pot' 'toykitchen6_sweet_potato_on_plate' 'toykitchen6_cucumber_in_pot' 'toykitchen6_knife_in_pot')
pixel_sep=(1)
encoder_type='resnet_34_v1'
proj_name=sanity_PTR

index=$1
dry_run=false
debug=false
single_gpu=false

echo "Running Experiment on ${tdatas[$index]}"


pip3 install immutabledict; 
pip3 install dill;
pip3 install gcsfs;
pip3 install protobuf==3.20.1;
pip3 install optax==0.1.2

export PYTHONPATH=~/jaxrl2/:$PYTHONPATH; 
export PYTHONPATH=~/m3ae_public/:$PYTHONPATH;
export EXP=~/hdd/$proj_name; 
export DATA=~/; 


rm -rf $EXP; 
mkdir -p $EXP; 


for tdata in ${tdatas[@]}; do
    for pixel in ${pixel_sep[@]}; do

        if [ $index -eq 0 ]; then
            echo "Launching $tdata with pixel separation $pixel"

            command="XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/launch_train_real_cql.py --prefix resnet34_all_pp_data --cql_alpha 10 \
            --encoder_type $encoder_type --algorithm cql_encodersep_parallel --dataset $data --target_dataset $tdata \
            --batch_size 64 --wandb_project $proj_name --multi_viewpoint 0 --add_prev_actions 0 --policy_encoder_type same \
            --target_mixing_ratio 0.9 --use_action_sep 1 --use_basis_projection 0 --discount 0.96 --max_q_backup 1 --basis_projection_coefficient 0.0 \
            --use_multiplicative_cond 0 --num_final_reward_steps 3 --term_as_rew 1 --encoder_norm group --use_spatial_learned_embeddings 1  \
            --target_entropy_factor 1.0 --policy_use_multiplicative_cond 0 --use_pixel_sep $pixel --use_normalized_features 0 --min_q_version 3 \
            --q_dropout_rate 0.0 --offline_finetuning_start 160000"

            if [ $debug = true ]; then
                if [ $single_gpu = true ]; then
                    echo "Using single GPU Debug"
                    command="$command --eval_interval 1 --log_interval 1 --debug 1"
                else
                    echo "Using multiple GPUs Debug"
                    command="$command --eval_interval 1  --log_interval 1"
                fi
            fi
            
            echo $command
            if [ $dry_run = false ]; then
                eval $command    
            fi
        fi
        
        index=$((index-1))
    done
done