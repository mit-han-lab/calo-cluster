for LR in 3e-1 6e-1 3e-2 6e-2 3e-3 6e-3
do
    python train.py dataset=hcal_Zll_jets_offset model=spvcnn_offset train=slurm dataset.event_frac=0.1 train.num_epochs=5 ~semantic_criterion embed_criterion=offset optimizer.lr=$LR wandb.name="hcal_lr$LR"
done