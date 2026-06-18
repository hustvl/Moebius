# Set WORK_DIR to your project root before running
THIS_SH_PATH=$CONFIG_FILE


OUTPUT_DIR='exp_outputs'
OUTPUT_DIR_EXP_NAME="${OUTPUT_DIR}/${EXP_NAME}"


export OUTPUT_DIR=$OUTPUT_DIR
export OUTPUT_DIR_EXP_NAME=$OUTPUT_DIR_EXP_NAME
export HF_HOME=$HF_HOME

export TRAIN_ARGS=" --data_type RemovalDataset_v1_2 \
                    --lognorm_t \
                    --elatentlpips_loss --elatentlpips_loss_weight 0.5 \
                    --task_loss         --task_loss_weight 0.5 \
                    --KD_loss_weight 0.01 \
                    --mse_feat_loss --feat_loss_weight 1.0 --feat_index_T 5 --feat_index_S 2 \
                    --model_config_path=config/model_cfg/moebius.yaml \
                    --teacher_weight_path=../../hf_models/hustvl/PixelHacker/pretrained/diffusion_pytorch_model.bin \
                    --teacher_config_path=config/model_cfg/pixelhacker.yaml \
                    --data_config=config/data_demo.yaml \
                    --num_embeddings 20 \
                    --image_size 512 \
                    --batch_size 2 \
                    --num_workers 4 \
                    --output_dir=${OUTPUT_DIR_EXP_NAME} \
                    --output_name=exp \
                    --seed=42 \
                    --learning_rate=1e-4 \
                    --global_step=0 \
                    --max_train_steps=200000 \
                    --save_every_n_steps=3000 \
                    --logging_dir=${OUTPUT_DIR_EXP_NAME}/log \
                    --gradient_accumulation_steps=1 \
                    --optimizer_type=Muon \
                    --lr_scheduler=constant_with_warmup \
                    --lr_warmup_steps=0 \
                    --save_precision=bf16 \
                    --mixed_precision=bf16 \
                    --noise_offset=0.0357 \
                    --gradient_checkpointing \
                    --xformers \
                    --log_with=tensorboard \
                    --script_args=$THIS_SH_PATH "
