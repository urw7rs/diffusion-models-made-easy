import pytorch_lightning as pl

from dmme.common import make_history, gaussian, denorm


class GenerateImage(pl.Callback):
    r"""Generate samples to check training progress

    Args:
        imgsize (Tuple[int, int, int]): A tuple of ints representing image shape :math:`(C, H, W)`
        batch_size (int): Number of samples to generate
        vis_length (int): Length of denoising sequence to visualize
        every_n_epochs (int): Only save those images every N epochs
        test (bool): generates images on test if set to true
    """

    def __init__(
        self, imgsize, batch_size=8, vis_length=20, every_n_epochs=5, test=False
    ):
        super().__init__()

        self.imgsize = imgsize
        self.batch_size = batch_size
        self.vis_length = vis_length
        self.every_n_epochs = every_n_epochs

        self.test = test

    def on_train_epoch_end(self, trainer, pl_module):
        self._shared_hook(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test:
            self._shared_hook(trainer, pl_module)

    def _shared_hook(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:

            if trainer.logger is None:
                return

            history = self.generate_img(pl_module)
            grid = make_history(history)

            if isinstance(trainer.logger, list):
                for logger in trainer.logger:
                    self._log(logger, grid)
            else:
                self._log(trainer.logger, grid)

    def _log(self, logger, grid):
        experiment = logger.experiment

        if isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image("generated_images", [grid])

        if isinstance(logger, pl.loggers.TensorBoardLogger):
            experiment.add_image("generated_images", grid, pl_module.global_step)

    def generate_img(self, pl_module):
        pl_module.eval()

        denoising_sequence = []

        x_t = gaussian((self.batch_size, *self.imgsize), device=pl_module.device)
        denoising_sequence.append(denorm(x_t))

        timesteps = pl_module.hparams.timesteps

        step = timesteps // (self.vis_length - 1)
        if timesteps % (self.vis_length - 1) > 0:
            step += 1

        t = timesteps
        save_t = t - step
        while t > 0:
            while t > save_t:
                x_t = pl_module(x_t, t, t - 1)
                t -= 1

            denoising_sequence.append(denorm(x_t.clone().detach()))

            save_t -= step

        pl_module.train()

        return denoising_sequence
