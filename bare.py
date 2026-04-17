from sugarcane_variety.colab_compatible import (
    print_eval_summary,
    run_all_for_colab,
    test_for_colab,
)


if __name__ == "__main__":
    prep, train = run_all_for_colab(
        raw_dir="content/data/raw", 
        prepared_dir="content/data/prepared",
        output_dir="content/data/sugarcane_artifacts",
        epochs=25,
        batch_size=32,
        image_size=224,
        workers=8,
        label_mode="variety_maturity",
        preprocess_device="cpu",
        preprocess_workers=8,
        perform_preprocess=False,
    )

    eval_result = test_for_colab(
        prepared_dir="content/data/prepared",
        checkpoint_path="content/data/sugarcane_artifacts/best_model.pt",
        batch_size=32,
        workers=8,
    )

    print(prep)
    print(train)
    print_eval_summary(eval_result)
