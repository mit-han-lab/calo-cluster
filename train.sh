for VX in 0.01 0.1 0.3 0.05 0.2 0.03 0.08
do
    python train.py dataset=hcal_Zll_jets_offset model=spvcnn_offset embed_criterion=offset dataset.train_frac=0.05 dataset.test_frac=0.9 train.num_epochs=5 ~semantic_criterion wandb.name="hcal_Zll_offset_vx$VX" model.embed_dim=2 cluster=slurm dataset.instance_target=truth dataset.voxel_size=$VX
done