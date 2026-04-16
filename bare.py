from sugarcane_variety.colab_compatible import run_all_for_colab, test_for_colab


if __name__ == "__main__":
    prep, train = run_all_for_colab(
        raw_dir="content/data/raw", 
        prepared_dir="content/data/prepared",
        output_dir="content/data/sugarcane_artifacts",
        epochs=25,
        batch_size=32,
        image_size=224,
        workers=8,
        preprocess_device="cpu",
        preprocess_workers=8,
    )

    eval_result = test_for_colab(
        prepared_dir="content/data/prepared",
        checkpoint_path="content/drive/MyDrive/sugarcane_artifacts/best_model.pt",
        batch_size=32,
        workers=8,
    )

    print(prep)
    print(train)
    print(eval_result)
