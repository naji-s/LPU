import lpu.datasets.LPUDataset
import lpu.models.sarpu_em
import lpu.utils.utils_general
import lpu.datasets.dataset_utils
TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5
def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/psychm_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    config['kernel_mode'] = 2
    config['max_iter'] = 10
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False)
    BATCH_SIZE = len(lpu_dataset)
    train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    sarpu_em_model = lpu.models.sarpu_em.SARPU(config, training_size=len(train_loader.dataset))
    sarpu_em_model.train(train_loader)
    print(sarpu_em_model.validate(val_loader))

if __name__ == "__main__":

    main()
