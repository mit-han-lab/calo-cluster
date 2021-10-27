for VX in 10.0 100.0
do
    python train.py dataset=hcal_Zll_jets_offset model=spvcnn_offset train=distributed dataset.event_frac=0.1 train.gpus=2 train.batch_size=4 dataset.num_workers=16 train.num_epochs=4 ~semantic_criterion embed_criterion=offset dataset.voxel_size=$VX wandb.name="hcal_vx$VX"
done