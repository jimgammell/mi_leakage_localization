

class TrialUtils:
    def __init__(self
    ):
        pass

    def train_supervised_classifier(self, logging_dir):
        data_module = self.data_module_constructor()
        data_module.setup('')
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
        training_module = self.training_module_constructor()
        checkpoint = ModelCheckpoint(filename='best', monitor='val-rank', save_top_k=1, mode='min')
        trainer = Trainer(
            max_epochs=self.epochs,
            default_root_dir=logging_dir,
            accelerator='gpu',
            devices=1,
            logger=TensorBoardLogger(logging_dir, name='lightning_output')
        )
        trainer.fit(training_module, datamodule=data_module)
        trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
        ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
        ea.Reload()
        training_curves = {
            key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-rank', 'val-rank']
        }
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
            pickle.dump(training_curves, f)
    
    def plot_classification_results(self, logging_dir):
        with open(training_curves_path, 'rb') as f:
            training_curves = pickle.load(f)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(*training_curves['train-loss'], color='red', linestyle='--')
        axes[0].plot(*training_curves['val-loss'], color='red', linestyle='-')
        axes[1].plot(*training_curves['train-rank'], color='red', linestyle='--')
        axes[1].plot(*training_curves['val-rank'], color='red', linestyle='-')
        axes[0].set_xlabel('Training step')
        axes[1].set_xlabel('Training step')
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Rank')
        fig.tight_layout()
        fig.savefig(os.path.join(logging_dir, 'training_curves.png'))
        plt.close('all')
    
    def run_lr_sweep(self, base_dir, learning_rates):
        for learning_rate in learning_rates:
            logging_dir = os.path.join(base_dir, f'learning_rate={learning_rate}')
            if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
                with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                    training_curves = pickle.load(f)
            else:
                self.train_supervised_classifier