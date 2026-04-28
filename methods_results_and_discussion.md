**3. Materials and Methods**

**3.1 Research Design**  
This study developed and evaluated VISCANE, an edge-based decision-support system for sugarcane quality grading and maturity detection. The system combined image-based classification with farmer-provided agronomic inputs to support harvest-readiness assessment, estimated yield computation, and recommendation generation. The work followed an applied system-development design, where the mobile application, backend services, image-classification model, and yield-estimation workflow were built and tested as one integrated prototype.

The development process followed an iterative Agile-Scrum approach. Each development cycle focused on a specific system function, such as image capture, model training, maturity classification, agronomic input collection, yield calculation, and report generation. This process allowed the system to be refined based on implementation testing and model-evaluation results.

**3.2 System Workflow**  
The VISCANE workflow began with the farmer capturing a sugarcane image through the mobile application. The system then processed the image for maturity and variety classification. The classification result was combined with farmer-entered agronomic data, including sugarcane variety, farm size, RSSI infestation severity, weeding frequency, fertilizer application frequency, ratoon stage, and plowing frequency.

After receiving these inputs, the system applied a deterministic yield-estimation process. The selected variety determined the baseline values and weighting factors used in the computation. The maturity classification affected the final estimate through maturity-based penalties. A Mature classification applied no maturity penalty, while Not Mature and Over Mature classifications reduced the estimated output according to the predefined variety-specific penalty values.

The final output included estimated LKG/TC, estimated LKG/HA, and farmer-facing recommendations. These recommendations were generated based on the predicted maturity status and the agronomic conditions entered by the user.

**3.3 Dataset Preparation**  
The image dataset was prepared according to sugarcane variety and maturity condition. The supported varieties were VMC 84-524, VMC 84-947, and MAURITIO RC888 / Mauritius RC888. The maturity classes were Not Mature, Mature, and Over Mature.

For model training, each image was assigned to a combined variety-and-maturity category. This allowed the model to learn both the sugarcane variety and maturity condition from the same image. The implemented categories included the following nine classes:

| Variety and Maturity Category | Variety | Maturity Class |
| --- | --- | --- |
| VMC 84-524, Mature | VMC 84-524 | Mature |
| VMC 84-524, Not Mature | VMC 84-524 | Not Mature |
| VMC 84-524, Over Mature | VMC 84-524 | Over Mature |
| VMC 84-947, Mature | VMC 84-947 | Mature |
| VMC 84-947, Not Mature | VMC 84-947 | Not Mature |
| VMC 84-947, Over Mature | VMC 84-947 | Over Mature |
| MAURITIO RC888 / Mauritius RC888, Mature | MAURITIO RC888 / Mauritius RC888 | Mature |
| MAURITIO RC888 / Mauritius RC888, Not Mature | MAURITIO RC888 / Mauritius RC888 | Not Mature |
| MAURITIO RC888 / Mauritius RC888, Over Mature | MAURITIO RC888 / Mauritius RC888 | Over Mature |

The dataset preparation followed a classification-based structure. Images were first organized by sugarcane variety and maturity status. The system then verified whether each image file could be opened properly. Corrupt or unreadable images were excluded from the prepared dataset. Exact duplicate files were grouped during preprocessing to reduce direct duplicate leakage, while near-duplicate captures were reviewed through the split-audit process.

After validation and grouping, the images were divided into training, validation, and testing sections. The training section was used to fit the classification models, the validation section was used to monitor model performance during training, and the testing section was reserved for final evaluation. In the documented training run, the prepared dataset contained 10,229 images: 3,416 for training, 3,404 for validation, and 3,409 for testing.

**Table 1. Dataset Distribution**

| Dataset Section | Purpose | Number of Images | Dataset Percentage |
| --- | --- | ---: | ---: |
| Training set | Used to train the image-classification models | 3,416 | 33.39% |
| Validation set | Used to monitor model performance during training | 3,404 | 33.28% |
| Testing set | Used for final model evaluation | 3,409 | 33.33% |
| Total prepared dataset | Complete dataset used in the study | 10,229 | 100% |

**3.4 Image Preparation and Augmentation**  
Before training, images were checked and prepared to ensure that the models received consistent inputs. For ResNet-18 classification, each training image was first resized to 256 x 256 pixels and then randomly cropped using `RandomResizedCrop` with a scale range of 0.95 to 1.00 and an aspect-ratio range of 0.95 to 1.05. The cropped image was then resized to the model input size of 320 x 320 pixels.

Data augmentation was applied only to the training set to improve the model’s ability to handle field-image variation. Because the dataset contains sugarcane stalk images, the augmentation settings were kept very conservative so that stalk shape, rind texture, and color cues were not strongly distorted. The ResNet-18 training transform used random horizontal flipping with a probability of 0.30, random rotation up to 3 degrees, and light color jitter with brightness 0.06, contrast 0.06, saturation 0.04, and no hue shift. Images were then converted to tensors and normalized using ImageNet mean values of [0.485, 0.456, 0.406] and standard deviation values of [0.229, 0.224, 0.225]. Validation and testing images were not augmented; they were resized directly to 320 x 320 pixels, converted to tensors, and normalized with the same ImageNet values. No random crop, random rotation, ColorJitter, or flipping was applied to the validation and testing sets.

For YOLOv8 classification, conservative augmentation settings were used or recommended to avoid unrealistic geometric and color changes: `hsv_h=0.001`, `hsv_s=0.10`, `hsv_v=0.10`, `degrees=2`, `translate=0.02`, `scale=0.10`, `shear=0.0`, `perspective=0.0`, `fliplr=0.3`, `flipud=0.0`, `mosaic=0.10`, `mixup=0.0`, `copy_paste=0.0`, and `close_mosaic=10`. The augmentation design avoided vertical flipping, 90-degree rotation, strong hue or saturation changes, heavy cropping, strong perspective transformation, heavy blur, heavy mosaic, and mixup.

The image workflow used RGB images as the main input representation. HSV-related adjustments were applied as part of the color augmentation process, particularly in the YOLOv8 classification training. In this implementation, HSV was used to support color variation during training rather than as a separate hand-crafted feature-extraction method.

**3.5 Image-Classification Models**  
Two image-classification model families were trained and evaluated: ResNet-18 and YOLOv8 classification. The recommended ResNet-18 setup uses a single output head that directly predicts the combined variety-maturity class. YOLOv8 was implemented using its classification configuration and was trained on the same prepared dataset.

The implemented YOLOv8 component was used for image classification. The evaluated project did not use YOLOv8 as a bounding-box detector or region-localization model. Therefore, the evaluated computer-vision task in this study was combined variety-maturity image classification.

Both models were trained for 45 epochs using a batch size of 32. ResNet-18 used a 320 x 320 input after the training transform described above. The learning rate was 0.0002 and the weight decay was 0.001. ResNet-18 also used class-weighted loss to reduce the effect of class imbalance during training. Balanced batch sampling was retained as an optional diagnostic setting. The purpose of training both model families was to determine which implemented classifier performed better for the VISCANE maturity-assessment workflow.

**3.6 Maturity Interpretation**  
The system interpreted each predicted category according to its variety and maturity components. For example, a prediction of VMC 84-524, Mature was interpreted as the VMC 84-524 variety with a Mature maturity status. The maturity component was then used by the recommendation and yield-estimation workflow.

When the model produced probability scores across the combined classes, the system derived the final maturity status by grouping the probabilities according to maturity class. The maturity class with the highest combined probability was selected as the final maturity result. This allowed the system to use combined variety-maturity classification while still producing a direct maturity output for farmers.

**3.7 Yield Estimation and Recommendation Process**  
The maturity result was combined with the agronomic inputs to estimate sugarcane quality and yield. The system used variety-based baseline values and weighting rules for RSSI infestation severity, weeding frequency, fertilizer application, ratoon stage, and plowing frequency.

The computation applied penalties when agronomic conditions were unfavorable. A maturity penalty was also applied when the image result was Not Mature or Over Mature. The resulting estimated values were expressed as LKG/TC and LKG/HA. The recommendation engine then used these outputs to generate guidance for harvest timing and field management.

Not Mature results triggered a recommendation to delay harvest and allow further maturity development. Mature results supported harvest-readiness decisions. Over Mature results triggered a recommendation to harvest urgently to reduce further quality loss. Agronomic issues such as RSSI infestation or insufficient field practice also contributed to the final recommendation.

**4. Results**

**4.1 Overall Model Performance**  
Both implemented classifiers produced high performance in the updated documented training run. ResNet-18 achieved a best validation accuracy of 97.69% and a final test accuracy of 97.54%. The YOLOv8 classification model achieved a best validation accuracy of 98.18% and a final test accuracy of 97.68%.

Both models were evaluated using 3,409 testing images and nine artifact classes. These results show that both implemented classifiers performed strongly on the prepared VISCANE image-classification dataset, with YOLOv8 producing a slightly higher final test accuracy and ResNet-18 producing similarly strong performance.

**Table 2. Overall Classification Performance**

| Model | Best Validation Accuracy | Test Accuracy | Testing Images | Number of Classes |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 97.69% | 97.54% | 3,409 | 9 |
| YOLOv8 Classification | 98.18% | 97.68% | 3,409 | 9 |

**4.2 Variety and Maturity Diagnostic Accuracy**  
The updated diagnostic results separated variety-recognition and maturity-recognition performance from the exact combined-label result. This helped determine whether the models were learning the variety component, the maturity component, or only the combined class label.

For ResNet-18, the variety-only diagnostic accuracy was 99.88%, while the maturity-only diagnostic accuracy was 97.54%. For YOLOv8, both the variety-only and maturity-only diagnostic accuracies were 97.68%. These results indicate that both models learned the visual patterns needed for variety and maturity recognition in the updated prepared dataset.

**Table 3. Diagnostic Accuracy by Evaluation Category**

| Model | Exact Combined-Label Accuracy | Variety-Only Accuracy | Maturity-Only Accuracy |
| --- | ---: | ---: | ---: |
| ResNet-18 | 97.54% | 99.88% | 97.54% |
| YOLOv8 Classification | 97.68% | 97.68% | 97.68% |

**4.3 Per-Class Performance**  
The ResNet-18 per-class results showed high performance across the nine classes. Eight classes achieved at least 98.78% accuracy, and six classes achieved 100.00% accuracy. The only notably lower class was MAURITIO RC888 / Mauritius RC888, Mature, which achieved 82.31% accuracy on 441 test images.

The lower result for MAURITIO RC888 / Mauritius RC888, Mature indicates that this class remained the main source of residual error for ResNet-18. Most other variety-maturity categories were classified with very few or no errors.

**Table 4. ResNet-18 Per-Class Accuracy**

| Variety and Maturity Category | Accuracy | Test Images |
| --- | ---: | ---: |
| VMC 84-524, Mature | 98.78% | 328 |
| VMC 84-524, Not Mature | 99.44% | 357 |
| VMC 84-524, Over Mature | 100.00% | 231 |
| VMC 84-947, Mature | 100.00% | 433 |
| VMC 84-947, Not Mature | 100.00% | 421 |
| VMC 84-947, Over Mature | 100.00% | 461 |
| MAURITIO RC888 / Mauritius RC888, Mature | 82.31% | 441 |
| MAURITIO RC888 / Mauritius RC888, Not Mature | 100.00% | 381 |
| MAURITIO RC888 / Mauritius RC888, Over Mature | 100.00% | 356 |

**4.4 Classification Errors**  
The remaining ResNet-18 confusion errors were concentrated in MAURITIO RC888 / Mauritius RC888, Mature. The largest error occurred when MAURITIO RC888 / Mauritius RC888, Mature images were predicted as MAURITIO RC888 / Mauritius RC888, Not Mature, with 58 cases. The second-largest error was MAURITIO RC888 / Mauritius RC888, Mature predicted as MAURITIO RC888 / Mauritius RC888, Over Mature, with 18 cases.

**Table 5. Major ResNet-18 Classification Errors**

| Actual Category | Predicted Category | Count |
| --- | --- | ---: |
| MAURITIO RC888 / Mauritius RC888, Mature | MAURITIO RC888 / Mauritius RC888, Not Mature | 58 |
| MAURITIO RC888 / Mauritius RC888, Mature | MAURITIO RC888 / Mauritius RC888, Over Mature | 18 |
| VMC 84-524, Mature | VMC 84-524, Not Mature | 2 |
| VMC 84-524, Mature | VMC 84-947, Not Mature | 2 |
| VMC 84-524, Not Mature | VMC 84-524, Mature | 2 |
| MAURITIO RC888 / Mauritius RC888, Mature | VMC 84-524, Over Mature | 1 |
| MAURITIO RC888 / Mauritius RC888, Mature | VMC 84-947, Not Mature | 1 |

**4.5 Dataset Quality and Diagnostic Findings**  
The updated dataset summary contained all nine expected variety-maturity classes and recorded no low-sample warnings for the configured threshold. Each class had several hundred images in the training, validation, and testing sections. This improved class coverage made the updated per-class results more stable than the earlier run.

Although the documented results were strong, split independence remains an important validation concern for image datasets collected from similar cane or stalk captures. Exact duplicate files were handled during preprocessing, but near-duplicate captures should still be reviewed with the split-audit output. This check is important because visually similar images across training, validation, and testing sections can inflate measured accuracy.

**4.6 System Output**  
The trained model output was integrated into the VISCANE decision-support workflow. The system used the predicted maturity status as one of the inputs for yield estimation and recommendation generation. When the model classified a sample as Mature, the system supported harvest-readiness recommendations. When the model classified a sample as Not Mature, the system advised delayed harvesting. When the model classified a sample as Over Mature, the system advised urgent harvesting to reduce possible quality loss.

The system also considered the agronomic inputs entered by the user. RSSI severity, farm size, weeding, fertilizer application, ratoon stage, and plowing frequency affected the final estimated LKG/TC and LKG/HA values. This allowed VISCANE to generate recommendations based on both visual maturity and field-management conditions.

**5. Discussion**

The results show that the VISCANE prototype successfully implemented an end-to-end process for image-based sugarcane classification, maturity interpretation, yield estimation, and farmer recommendation. In the updated documented run, both ResNet-18 and YOLOv8 classification achieved high accuracy on the prepared dataset. YOLOv8 achieved the highest final test accuracy at 97.68%, while ResNet-18 achieved 97.54%.

The variety-only and maturity-only diagnostic results also improved substantially. ResNet-18 achieved 99.88% variety-only accuracy and 97.54% maturity-only accuracy. YOLOv8 achieved 97.68% for both diagnostic categories. These results indicate that the updated dataset and training configuration allowed both models to learn the main visual distinctions required by the VISCANE workflow.

The remaining ResNet-18 errors were concentrated in MAURITIO RC888 / Mauritius RC888, Mature. Most of these errors were predictions into the same variety but a different maturity status, especially Not Mature and Over Mature. This suggests that the remaining weakness is not broad variety confusion but fine-grained maturity separation within MAURITIO RC888 / Mauritius RC888.

The class distribution was also improved compared with the earlier training run. The updated documented dataset included all nine expected variety-maturity classes, including MAURITIO RC888 / Mauritius RC888, Not Mature. No low-sample warnings were recorded, and each class had hundreds of test images. This makes the updated performance values more meaningful than earlier results that had missing or very small classes.

However, the high accuracy should still be interpreted together with split-quality evidence. If near-duplicate images from the same cane or stalk appear across training, validation, and testing sections, the reported validation and test performance may overestimate field generalization. Future experiments should include cane- or stalk-level split auditing and, if possible, an additional independent field-test set collected separately from the training images.

The implemented project also clarified the actual role of YOLOv8 in the system. In the evaluated version, YOLOv8 was trained as an image-classification model, not as a detection model. Therefore, the current research results should describe YOLOv8 as a classification baseline within the project. If a future version adds sugarcane stalk detection or localization, that version should be evaluated separately using detection-specific outputs.

Overall, the project demonstrates that VISCANE can combine image classification with deterministic agronomic computation to support sugarcane quality assessment. The system workflow is functional, and the updated results show strong classification performance on the prepared dataset. Future work should focus on confirming split independence, validating the model on a separately collected field-test set, reviewing the remaining MAURITIO RC888 / Mauritius RC888 Mature confusion cases, validating maturity labels with domain experts, and standardizing field-image capture.

**Research Inputs Still Needed Before Final Submission**

| Needed Input | Purpose |
| --- | --- |
| Split leakage audit summary | Confirm whether exact or near-duplicate captures remain across splits |
| Independent field-test result | Confirm that high accuracy generalizes beyond the prepared dataset |
| Review of MAURITIO RC888 / Mauritius RC888 Mature errors | Determine whether remaining errors are visual ambiguity, label noise, or model weakness |
| Domain-expert label validation | Confirm maturity labels for visually ambiguous samples |
