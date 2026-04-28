**3\.  Materials and methods**

**3.1 System Methodology.**   
The researchers adopted the Agile-Scrum Software Development Life Cycle (SDLC) as the approach for building the system. Agile-Scrum is an iterative and incremental approach to software development that equips flexibility, collaboration, and client satisfaction. It allows requirements and solutions to progress through teamwork, stakeholder involvement, and continuous refinement. By breaking development into short sprint cycles, the model supports the delivery of functional components, frequent feedback, and adaptability to changing project needs.  
Within this framework, a strong significance was placed on computer vision for detecting sugarcane maturity. The system integrated YOLOv8 for sugarcane detection and localization and ResNet-18 for maturity classification. The models were prepared as separate computer vision components and integrated into the image-analysis workflow so that localized sugarcane regions could be evaluated for maturity. Each sprint delivered functional modules, including image processing, agronomic data entry, prediction models, and system integration.

	  
**Figure 1** Agile-Scrum System Development (SDLC)

 **3.2 System Development and Architecture.**   
The system was built to establish efficiency, accessibility, and reliable performance in remote farming areas. It uses modern software and AI frameworks and features a dual-interface design: a web portal for administrators and a mobile app for farmers. Its primary goal is to simplify sugarcane quality grading and provide estimated yield forecasts. By embedding edge computing in the mobile platform, farmers can perform real-time maturity checks in the field, even with poor connectivity.  
Through this setup, VISCANE equips smallholder farmers with better harvest decisions and improved perceptions of outcomes. This will also give them guidance in planning more effectively for post-harvest processes. The mobile app runs computer vision models locally to present instant analysis of sugarcane maturity from RGB-HSV images.   
Development used the following tools and technologies:

* Operating System – Windows 11 and macOS Tahoe 26 as the primary environment. Linux Ubuntu 22 for Training and Inference server.  
  * Collaboration – GitHub, Google Drive, and Facebook Messenger for file sharing and communication.  
  * Documentation – Microsoft Word and Google Docs for drafting and formatting.  
  * Development Platforms – Visual Studio Code and Docker for backend testing, Android Studio for mobile deployment.  
  * Front-end – HTML, CSS, and JavaScript via Flask templating.  
  * Back-end – Python 3.14 with Flask, integrated with scikit-learn for regression modeling.  
  * Mobile Development – Android Studio with edge-AI inference.  
  * Computer Vision – OpenCV for color conversion, YOLOv8 for sugarcane detection/localization, and ResNet-18 for maturity classification.  
  * Database – PostgreSQL for relational storage, user management, and agronomic logging.  
  * Dataset Training – Google Colab for training and experimentation with machine learning datasets.

**3.3 Agronomic Parameterization and Deterministic Modeling**  
To ensure that yield predictions remain consistent with SRA-based baseline assumptions and resistant to degradation from sparse or noisy crowdsourced data (the "cold start" problem), the predictive engine uses a deterministic, weighted-baseline approach. The system is parameterized with localized yield metrics established by the Sugar Regulatory Administration (SRA) for highly prevalent sugarcane varieties in Isabela, Negros Occidental, specifically VMC 84-524, VMC 84-947, and MAURITIO RC888.  
In the backend architecture, a strict mapping function is implemented using the scikit-learn library. A Linear Regression model, configured with intercept suppression (fit\_intercept \= False), is dynamically fitted to a predefined matrix of varietal weights. These weights correspond to five critical agronomic inputs: Red-Striped Soft Scale (RSSI) pest infection severity, weeding frequency, fertilizer application frequency, ratoon stage, and plowing frequency. When a farmer submits a scan, their specific field inputs xk are evaluated against the variety-specific weights wk via this regression model to compute a continuous agronomic adjustment variable *(A)*.

**3.4 Yield Prediction and Quality Grading Algorithms**  
The final algorithmic grading relies on the mathematical fusion of the edge-derived maturity classification and the cloud-computed agronomic variables. The computer vision module returns a maturity status from the ResNet-18 classifier. In the implementation, ResNet-18 produces class logits that are converted to softmax probabilities ordered according to the saved class list. For maturity-only models, the predicted class corresponds directly to the maturity label. For combined variety-maturity labels, the backend derives the maturity status by summing the probabilities of all classes with the same decoded maturity label and selecting the maturity category with the highest marginal probability.

The preserved maturity classes are:

* Not Mature
* Mature
* Over Mature

Simultaneously, the regression engine calculates the raw Agronomic Adjustment *(A)* and isolates negative field-management practices into an Agronomic Penalty *(P).* Crucially, to account for physiological yield losses from suboptimal harvest timing, a variety-specific maturity penalty wmaturityv derived from the computer vision classification is integrated into the penalty calculation. This ensures that yield deductions caused by sucrose inversion in over-mature cane or excessive stalk moisture in not mature cane are mathematically enforced:

A = kwk ×xk  
P = k(0,wk ×xk) + wmaturityv

To prevent the computation of negative theoretical yields, the penalty is converted into a bounded Agronomic Multiplier (M): 

M \=0, 1 \+ P 

The final predictive yield outputs (y) are computed by scaling the official SRA Baselines (*B)* using the multiplier *(M).* These functions strictly bound the predictions, ensuring they do not exceed the theoretical maximums for the selected variety *(v),* ratoon stage *(s),* and farm size in hectares *(h)*:

yLKGTC=max0, minBLKGTCv,s, BLKGTCv,sM   
yTCHA= max0, minBTCHAv,sh, BTCHAv,sh ×M

Ultimately, the system outputs the final Predicted Quality Grade by directly fusing the AI maturity assessment with the regression-based agronomic adjustment:

Predicted Quality Grade \= Maturity Classification Result \+ A   
**3.5 Varietal Parameterization and Agronomic Weighting Algorithms**  
The system predictive engine computes the overarching Agronomic Adjustment (*A*) through a deterministic linear combination of five distinct farmer inputs (*x*): Red-Stripped Soft Scale (RSSI) pest infection severity, weeding frequency, fertilizer application frequency, ratoon stage, and plowing frequency. Each input variable is mathematically scaled by a predefined, variety-specific agronomic weight (*w*) to accurately reflect the unique biological resilience and cultivation requirements of the selected cultivar.  
The general algorithmic structure for the agronomic adjustment is defined as:

A \= wrssixrssi+ wweedingxweeding+ wfertilizerxfertilizer+ wratoonxratoon+ wplowingxplowing  
To accommodate the localized agricultural landscape of the target region, the system is fundamentally parameterized for three highly prevalent sugarcane varieties. The specific predictive algorithms and empirical weight distributions established for each variety are detailed below:

**VMC 84-524**   
This cultivar demonstrates a strong positive yield response to frequent weeding and precise fertilizer application but exhibits moderate sensitivity to yield degradation during ratoon aging. As a cultivar with a standard ripening curve, it receives a baseline maturity penalty of \-0.20 for premature harvesting due to unformed sugars and \-0.15 for delayed harvesting. The specific adjustment algorithm is formulated as:

AVMC84*\-*524=-0.45⋅xrssi+0.35⋅xweeding+0.25⋅xfertilizer+-0.12⋅xratoon+0.08⋅xplowing

**VMC 84-947**  
Characterized as a fast-growing cultivar, VMC 84-947 requires less intensive weeding and plowing interventions compared to VMC 84-524. Notably, it demonstrates excellent biological resilience across multiple ratoon stages, which is reflected mathematically by its minimized negative weight for ratoon aging. However, because fast-growing cultivars experience highly accelerated sucrose inversion post-peak, it incurs a severe \-0.22 penalty when classified as over-mature. The specific adjustment algorithm is formulated as:

AVMC 84\-947=-0.45xrssi+0.28xweeding+0.18xfertilizer+-0.05xratoon+0.05xplowing

**MAURITIO RC888 (Mauritius RC888)**   
While this variety responds moderately well to standard cultivation practices, it is highly susceptible to foliar diseases. Consequently, the predictive algorithm enforces a more severe negative penalty weight when RSSI infection is present. Furthermore, because this cultivar is slow to accumulate initial Brix, it suffers a heavy \-0.22 penalty if harvested prematurely. The specific adjustment algorithm is formulated as:

AMAURITIORC888=-0.55xrssi+0.28xweeding+0.18xfertilizer+-0.12xratoon+0.08xplowing

**3.5.1 Summary of Agronomic Weights**   
Table 1 outlines the comparative distribution of the predefined agronomic weights (wk) assigned to field inputs, as well as the dynamic maturity penalties (wmaturityv) triggered by the computer vision classification across the supported cultivars. 

**Table 1** Distribution of Variety-Specific Agronomic Weights *(w)*

| Agronomic Input Variable *(x)* | VMC 84-524 Weight (*w)* | VMC 84-947 Weight *(w)* | MAURITIO RC888 Weight *(w)* |
| :---: | :---: | :---: | :---: |
| RSSI (Disease Severity) | \-0.45 | \-0.45 | \-0.55 |
| Weeding Frequency | 0.35 | 0.28 | 0.28 |
| Fertilizer Application | 0.25 | 0.18 | 0.18 |
| Ratoon Stage | \-0.12 | \-0.05 | \-0.12 |
| Plowing Frequency | 0.08 | 0.05 | 0.08 |
| CV Classification |  |  |  |
| Maturity: Not Mature | \-0.20 | \-0.18 | \-0.22 |
| Maturity: Mature | 0.00 | 0.00 | 0.00 |
| Maturity: Over Mature | \-0.15 | \-0.22 | \-0.18 |

**3.6 Automated Decision Support and Recommendation Engine**  
Beyond data prediction, the backend functions as an automated agronomic advisory tool. Upon calculation of the final metrics, the DSS evaluates the predicted estimated LKG/TC against an 85% baseline threshold. If the prediction falls below this threshold, the algorithm parses the negative vectors within the Agronomic Penalty (P) to generate targeted, real-time interventions. For instance, the detection of severe RSSI dynamically triggers automated alerts detailing the PHILSURIN chemical control protocols, while sub-optimal cultivation inputs generate corrective 2-time or 3-time fertilizer and weeding schedules based on varietal requirements. Furthermore, the Recommendation Engine provides immediate harvest scheduling directives based on the computer vision maturity classification. Stalks classified as "Not Mature" prompt advisories to delay cutting to allow for optimal sucrose accumulation; "Mature" classifications trigger recommendations to finalize immediate harvest and transport logistics; and "Over Mature" results issue urgent alerts to expedite harvesting to mitigate further yield degradation caused by sucrose inversion.

            **3.7 Datasets**  
This section should provide enough detail to allow full replication of the study by suitably skilled investigators. Protocols for new methods should be included, but well-established protocols may simply be referenced. The reviewed project code expects a raw image dataset organized by sugarcane variety and maturity status using a directory structure equivalent to `raw_dir/<variety_name>/<maturity_status>/*.jpg`. The preprocessing pipeline supports `variety`, `maturity`, and `variety_maturity` label modes. For the combined setting, labels are encoded as `<variety>__<maturity_status>` and later decoded by the backend for variety and maturity reporting. The default split ratios in the training interface are 70% training, 15% validation, and 15% testing, using group-aware splitting to reduce duplicate or near-duplicate leakage across splits.

**Table 2** Datasets

| Sugarcane Varieties and Maturity | Sample Images | Training Images | Testing Images | Validation Images | Total Images |
| ----- | ----- | ----- | ----- | ----- | ----- |
|  VMC 84-524 (Not Mature) | [TBD] | [TBD] | [TBD] | [TBD] |  1,000 |
|  VMC 84-524 (Mature) | [TBD] | [TBD] | [TBD] | [TBD] |  1,000 |
|  VMC 84-524 (Over Mature)  | [TBD] | [TBD] | [TBD] | [TBD] |  1,000 |
|  VMC 84-947 (Not Mature)  | [TBD] | [TBD] | [TBD] | [TBD] |  1,000 |
|  VMC 84-947 (Mature)  | [TBD] | [TBD] | [TBD] | [TBD] |  1,000 |
| VMC 84-947 (Over Mature)  | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MAURITIO RC888 / Mauritius RC888 (Not Mature) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MAURITIO RC888 / Mauritius RC888 (Mature) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MAURITIO RC888 / Mauritius RC888 (Over Mature) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**3.8 Computer Vision Pipeline**  
The reviewed project source code implements the computer vision component as a supervised image-classification pipeline for sugarcane variety and maturity recognition. The code supports two trainable model types, `resnet18` and `yolov8`, and compares them using the same prepared dataset. In this implementation, YOLOv8 is configured through Ultralytics using the classification base weights `yolov8n-cls.pt`; the reviewed code and the training report do not show a YOLOv8 bounding-box detector, ROI localization output, detector confidence threshold, or non-maximum suppression configuration. Therefore, Section 3.8 describes YOLOv8 as a classification model in the implemented project, while ResNet-18 is described as the primary convolutional classifier built with PyTorch and torchvision.

**3.8.1 Dataset Preparation and Image Pre-processing**  
The dataset preprocessing module expects the raw image directory to be organized by variety and maturity status using the structure `raw_dir/<variety_name>/<maturity_status>/*.jpg`. The preprocessing code supports three label modes: `variety`, `maturity`, and `variety_maturity`. The training run used `variety_maturity`, which preserves combined labels for each variety and maturity group. These labels are later decoded by the backend into separate `variety` and `maturity_status` fields.

Before training, the preprocessing pipeline verifies image files, removes corrupt entries, groups duplicate or near-duplicate captures to reduce split leakage, and writes the prepared images into `train`, `val`, and `test` folders. The training script used a preprocessing resize value of 384 pixels and a model input size of 320 pixels. During preprocessing, images are converted to RGB and center-square cropped before resizing. During model training, additional augmentation is applied. ResNet-18 uses random resized cropping, horizontal flipping, rotation, color jitter, Gaussian blur, Gaussian noise, and random erasing. YOLOv8 uses Ultralytics augmentation parameters, including rotation, translation, scaling, left-right flipping, up-down flipping, HSV hue adjustment, HSV saturation adjustment, HSV value adjustment, and random erasing.

**3.8.2 ResNet-18 Classification Model**  
The ResNet-18 model is implemented using `torchvision.models.resnet18`. The final fully connected layer is replaced with a linear layer whose output dimension matches the number of dataset classes. During training, the model is initialized with ImageNet-pretrained weights, and the saved checkpoint stores the model type, model state dictionary, class labels, and image size. The implementation exports Android-oriented artifacts when possible, including PyTorch Lite and ONNX files.

For inference, uploaded images are converted to RGB, represented as tensors, resized by the shorter edge to 1.15 times the model image size, center-cropped to the saved image size, scaled to the 0.0 to 1.0 range, and normalized using ImageNet mean values of 0.485, 0.456, and 0.406 and standard deviation values of 0.229, 0.224, and 0.225. The model returns class logits, which are converted to softmax probabilities. The output is therefore a probability distribution over the saved class list, not a manually defined visual-feature array.

The training configuration used for the reported ResNet-18 run was 45 epochs, batch size of 32, image size of 320 pixels, learning rate of 0.0002, weight decay of 0.001, seed value of 42, label smoothing of 0.10, four frozen-backbone epochs, Gaussian noise standard deviation of 0.02, blur probability of 0.05, random-erasing probability of 0.05, and rotation range of 8 degrees. The `TrainingRes.pdf` report recorded a best validation accuracy of 59.57%, test accuracy of 59.91%, variety accuracy of 81.00%, and maturity accuracy of 64.70% for ResNet-18 across 3,547 test samples and 8 loaded classes.

**3.8.3 YOLOv8 Classification Model**  
The YOLOv8 model is implemented with the Ultralytics `YOLO` interface and trained using the base weights `yolov8n-cls.pt`. The source code trains YOLOv8 on the same prepared image-classification folder structure used by ResNet-18. The model is trained with the `imgsz` parameter set from the project configuration, and the reported run used an image size of 320 pixels. The YOLOv8 training configuration used 45 epochs, batch size of 32, learning rate of 0.0002, weight decay of 0.001, seed value of 42, translation of 0.12, scale of 0.25, left-right flip probability of 0.50, up-down flip probability of 0.12, HSV hue adjustment of 0.02, HSV saturation adjustment of 0.75, HSV value adjustment of 0.45, and random erasing of 0.35.

The YOLOv8 classifier is saved as an Ultralytics checkpoint at `content/data/sugarcane_artifacts/yolov8/yolov8/weights/best.pt`, with ONNX export attempted for deployment. During inference, the API loads the YOLOv8 checkpoint and reads the model class names from `model.names`. Prediction returns class probabilities, and the backend extracts the top-ranked classes with confidence values. The `TrainingRes.pdf` report recorded a best validation accuracy of 54.12% and a test accuracy of 56.19% for YOLOv8 across the same 3,547 test samples and 8 loaded classes. The report also lists generated YOLOv8 training artifacts, including confusion matrix images, results plots, training batches, and validation prediction images.

**3.8.4 Class Labels and Maturity Extraction**  
The source code defines known maturity labels as `MATURE`, `NOT_MATURE`, and `OVER_MATURE`. For combined variety-maturity training, class names are encoded using the format `<variety>__<maturity_status>`. The backend decodes this format into a variety component and a maturity-status component. When generating the maturity result, the backend sums softmax probabilities for all classes that share the same decoded maturity label and selects the maturity category with the highest aggregated probability. This allows the system to report maturity even when the classifier is trained on combined variety-maturity labels.

The reported ResNet-18 class list contains 8 loaded classes: `524__MATURE`, `524__NOT_MATURE`, `524__OVER_MATURE`, `847__MATURE`, `847__NOT_MATURE`, `847__OVER_MATURE`, `Mauritio__MATURE`, and `Mauritio__OVER_MATURE`. The available report does not show a `Mauritio__NOT_MATURE` class among the loaded classes, so this class remains a dataset-completion item if it is required by the study design.

**3.8.5 Model Selection and Backend Integration**  
The backend API loads both supported model types when checkpoints are available and selects the preferred model from the saved training report when possible. The `/predict` endpoint accepts an uploaded image, runs prediction for each loaded model, and returns the top class, confidence value, decoded variety, and decoded maturity status. The `/predict/annotated` endpoint uses the ResNet-18 model when available, otherwise YOLOv8, and returns an annotated image with maturity information in the response headers.

The classification result is then used by the broader VISCANE decision-support workflow. The maturity status is combined with the selected variety and agronomic inputs, including RSSI infestation severity, weeding frequency, fertilizer application frequency, ratoon stage, plowing frequency, farm size, and SRA-based baseline values. These inputs are used by the deterministic yield-prediction and recommendation modules to compute bounded LKG/TC and LKG/HA estimates, apply variety-specific maturity penalties, and generate harvest or field-management advisories.

**Summary of Changes**

* Revised Section 3.8 to match the reviewed source code and `TrainingRes.pdf`.
* Removed the unsupported YOLOv8 localization/ROI handoff description from Section 3.8.
* Described YOLOv8 as the implemented Ultralytics classification pipeline using `yolov8n-cls.pt`.
* Retained ResNet-18 as the PyTorch classification model and added the reported training configuration.
* Added reported training outcomes for ResNet-18 and YOLOv8 without inventing unreported detector metrics.

**Remaining [TBD] Fields for Researcher Input**

* Confirm whether the final manuscript should still claim YOLOv8 detection/localization, because the reviewed source code and `TrainingRes.pdf` show YOLOv8 classification rather than bounding-box detection.
* Confirm whether `Mauritio__NOT_MATURE` exists in the final dataset, because it is not listed among the 8 loaded classes in `TrainingRes.pdf`.
* Dataset sample, training, testing, and validation counts for each variety-maturity category.
* Total image counts for VMC 84-947 (Over Mature) and MAURITIO RC888 / Mauritius RC888 classes.
