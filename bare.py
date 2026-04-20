from sugarcane_variety.colab_compatible import (
    print_eval_summary,
    run_all_for_colab,
    test_for_colab,
)


if __name__ == "__main__":
    prep, resnet_train = run_all_for_colab(
        raw_dir="content/data/raw/DATASETSFINAL", 
        prepared_dir="content/data/prepared",
        output_dir="content/data/sugarcane_artifacts/resnet18",
        epochs=25,
        batch_size=32,
        image_size=224,
        workers=8,
        label_mode="variety_maturity",
        preprocess_device="cpu",
        preprocess_workers=8,
        perform_preprocess=True,
        model_type="resnet18",
    )

    yolo_prep, yolo_train = run_all_for_colab(
        raw_dir="content/data/raw/DATASETSFINAL",
        prepared_dir="content/data/prepared",
        output_dir="content/data/sugarcane_artifacts/yolov8",
        epochs=25,
        batch_size=32,
        image_size=224,
        workers=8,
        label_mode="variety_maturity",
        preprocess_device="cpu",
        preprocess_workers=8,
        perform_preprocess=False,
        model_type="yolov8",
        yolo_weights="yolov8n-cls.pt",
    )

    resnet_eval = test_for_colab(
        prepared_dir="content/data/prepared",
        checkpoint_path=resnet_train.checkpoint_path,
        batch_size=32,
        workers=8,
        model_type="resnet18",
    )
    yolo_eval = test_for_colab(
        prepared_dir="content/data/prepared",
        checkpoint_path=yolo_train.checkpoint_path,
        batch_size=32,
        workers=8,
        model_type="yolov8",
    )

    print(prep)
    print(yolo_prep)
    print(resnet_train)
    print(yolo_train)
    print("ResNet18 evaluation")
    print_eval_summary(resnet_eval)
    print("YOLOv8 evaluation")
    print_eval_summary(yolo_eval)
