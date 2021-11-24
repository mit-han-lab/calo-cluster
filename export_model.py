
from calo_cluster.evaluation.experiments.hcal_pf_experiment import \
    HCalPFExperiment


def main():
    exp = HCalPFExperiment(wandb_version='3bybg4l1')
    del exp.model.instance_criterion
    dataloader = exp.datamodule.val_dataloader()
    feats = dataloader.dataset[0]['features']
    onnx_path = exp.ckpt_path.parent.parent / 'model.onnx'
    exp.model.to_onnx(file_path=onnx_path, input_sample=feats, export_params=True)
    #script = exp.model.to_torchscript(example_inputs=feats)

if __name__ == "__main__":
    main()