import os
import sys

# Add libs to path for dinov2 and ADE20K modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(SCRIPT_DIR, 'libs')
sys.path.insert(0, LIBS_DIR)

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.mode == 'train':
        import wandb
        from trainer_ce import TrainerCE
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerCE(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    elif cfg.mode == 'train_class':
        import wandb
        from trainer_class import TrainerClass
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerClass(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    elif cfg.mode == 'train_instance':
        import wandb
        from trainer_instance import TrainerInstance
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerInstance(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    
    elif cfg.mode == 'visualize':
        from visualizer import Visualizer
        visualizer = Visualizer(cfg, cfg.output_dir)
        visualizer.vis()
    elif cfg.mode == 'eval_all':
        from trainer_ce import TrainerCE
        trainer = TrainerCE(cfg, cfg.output_dir)
        trainer.eval_all(cfg)
    
    elif cfg.mode == 'experiment_distance':
        from experiment_distance import DistanceExperiment
        experiment = DistanceExperiment(cfg)
        experiment.run()

    
if __name__ == "__main__":
    main()