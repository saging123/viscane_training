**VISCANE: An Edge-Based AI for Sugarcane Quality Grading and Maturity Detection Using RGB-HSV Imaging** 

**Faye N. Cuevas\*, Windy D. Elipse, John Rolo C. Flores,**   
**Lynol I. Ibarra**  
College of Information and Communications Technology, STI West Negros University, Bacolod City, Philippines  
Email: [cuevas.247492@wnu.sti.edu.ph](mailto:cuevas.247492@wnu.sti.edu.ph)

**Abstract (12-point bold font on single line spacing)**  
For abstract content, use 12-point Times New Roman font on single line spacing. First line is indented 0.5 inch.  An abstract of up to **250 words** must be included. Include your major findings in a useful and concise manner. Include a problem statement, objectives, brief methods, results, and the significance of your findings.

***Keywords*****:** *List up to 5 keywords and separate each keyword by a semi-colon (;).  The keywords should accurately reflect the content of the article.  The keywords will be used for indexing purposes.*

# 

# **1\.  Introduction**

Sugarcane (Saccharum officinarum) is one of the world’s most important crops. It produces over 1.9 billion metric tons annually. It is a major source of sugar, bioethanol, and biomass energy (Food and Agriculture Organization of the United Nations, 2025). According to Rose and Chilvers (2018), farmers face challenges such as climate change, resource scarcity, and market instability. This led to the adoption of digital tools, such as artificial intelligence and precision farming, to improve productivity and sustainability.   
In Southeast Asia, sugarcane supports rural employment and regional growth, but it still faces pressures from changing weather, labor shortages, and weak supply chains (Solomon et al., 2024). Countries such as Thailand, the Philippines, and Indonesia are adopting technologies, namely satellite imaging, mobile advisory services, and blockchain systems, to improve transparency (Diyanah, Kozono, and Hazmi, 2024). Despite these advances, rural adoption remains low due to poor infrastructure and limited digital skills (Montesclaros and Teng, 2023).  
In the Philippines, sugarcane covers more than 400,000 hectares, with 85% of farmers cultivating plots of five hectares or less (Sugar Regulatory Administration \[SRA\], 2019). This smallholder dominance leaves the industry vulnerable to unstable yields and market price volatility. Programs like block farming and digital agriculture have shown potential to increase productivity and incomes (Pantoja et al., 2019; Briones et al., 2023). However, adoption remains limited due to the need for inclusive policies and cooperative systems (Oñal, Junitilla, and Ortega, 2021).  
Negros Occidental, known as the “Sugar Bowl of the Philippines,” produces more than half of the country’s sugar. Yet smallholder farmers continue to struggle with declining yields, rising costs, and climate risks (Briones, Galang, and Latigar, 2023). Reliance on traditional practices and limited access to digital tools reduce competitiveness, which makes technology adoption essential to sustain livelihoods and protect the province’s role in the national sugar industry (SRA, 2019).   
To address these challenges, this study introduces VISCANE: An Edge-Based AI System for Sugarcane Quality Grading and Maturity Detection Using RGB-HSV Imaging.   
VISCANE integrates farmer-reported inputs with image-based maturity classification to generate real-time estimated predictions of sugar content. Specifically, the system employs YOLOv8 for real-time detection of sugarcane stalks and ResNet for deep feature extraction and maturity-level classification. By combining these models, VISCANE achieves both speed and accuracy in computer vision analysis and ensures reliable grading outcomes.  
By means of offering a low-cost, non-destructive, and accessible tool, the system empowers smallholder farmers, guides them in assessing sugarcane quality before and after harvest, and strengthens their participation in the sugarcane value chain. This research contributes to inclusive digital transformation and aligns with national and ASEAN priorities for resilience and competitiveness.

**2\.  Objectives**  
The main objective of this study is to develop VISCANE, an edge-based AI system for sugarcane quality grading and maturity assessment using RGB-HSV imaging. The system is designed to provide smallholder farmers with a non-destructive, accessible decision-support tool that improves grading accuracy and guides farmers in evaluating the quality of sugarcane before and after harvesting.  
Specifically, the study aims to:

* Design and implement an edge-based AI architecture capable of processing RGB-HSV imaging data for sugarcane quality and maturity grading.  
* Establish a non-destructive imaging methodology that guarantees transparency, operational reliability, and ease of use for smallholder farmers operating in rural environments.  
*  Integrate beneficiary-centered features into the system to support equitable valuation and reliable crop assessment, specifically through:  
  1. Inclusion of essential agronomic inputs, such as crop variety, total land area, and infestation severity;  
  2. Application of computer vision for maturity detection to facilitate objective crop grading (e.g., classifying stalks as “Mature”);  
  3. Utilization of model-based yield calculations anchored on baseline LKG/TC and LKG/HA data from the Sugar Regulatory Administration (SRA), mathematically adjusted for agronomic penalties (e.g., a localized penalty for RSSI infestation);  
  4. Computation of area-weighted LKG/TC and LKG/HA averages to project realistic, farm-level production outcomes;  
  5.  Generation of predictive estimated yield outputs that translate the user's input land area (in hectares) into actionable Total LKG forecasts (e.g., 25 tons × 1.87 LKG/TC \= 46.75 LKG); and  
  6.  Development of transparent reporting interfaces that display these calculations in a clear, accessible format to guide farmer evaluation during pre- and post-harvest stages.  
* Evaluate the system’s effectiveness and efficiency in both pre-harvest and post-harvest settings, ensuring alignment with local realities, improved decision-making, and farmer empowerment through accessible and actionable insights.


**3\.  Materials and methods**

**3.1 System Methodology.**   
The researchers adopted the Agile-Scrum Software Development Life Cycle (SDLC) as the approach for building the system. Agile-Scrum is an iterative and incremental approach to software development that equips flexibility, collaboration, and client satisfaction. It allows requirements and solutions to progress through teamwork, stakeholder involvement, and continuous refinement. By breaking development into short sprint cycles, the model guarantees rapid delivery of functional components, frequent feedback, and adaptability to changing project needs.  
Within this framework, a strong significance was placed on computer vision for detecting sugarcane maturity. The system integrated YOLOv8 for object detection and ResNet for image classification, both of which were trained and validated separately to ensure accuracy. Their outputs were combined to improve grading performance, which integrates a reliable RGB-HSV image analysis for sugarcane quality assessment. Each sprint delivered functional modules, including image processing, agronomic data entry, and prediction models, with the system embedded throughout.

	  
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
  * Computer Vision – OpenCV for color conversion, YOLOv8 for detection, ResNet-50 for classification.  
  * Database – PostgreSQL for relational storage, user management, and agronomic logging.  
  * Dataset Training – Google Colab for training and experimentation with machine learning datasets.

**3.3 Agronomic Parameterization and Deterministic Modeling**  
To ensure that yield predictions remain scientifically valid and resistant to degradation from sparse or noisy crowdsourced data (the "cold start" problem), the predictive engine uses a deterministic, weighted-baseline approach. The system is parameterized with localized yield metrics established by the Sugar Regulatory Administration (SRA) for highly prevalent sugarcane varieties in Isabela, Negros Occidental, specifically VMC 84-524, VMC 84-947, and MAURITIO RC888.  
In the backend architecture, a strict mapping function is implemented using the scikit-learn library. A Linear Regression model, configured with intercept suppression (fit\_intercept \= False), is dynamically fitted to a predefined matrix of varietal weights. These weights correspond to five critical agronomic inputs: Red-Striped Soft Scale (RSSI) pest infection severity, weeding frequency, fertilizer application frequency, ratoon stage, and plowing frequency. When a farmer submits a scan, their specific field inputs xk are evaluated against the variety-specific weights wk via this regression model to compute a continuous agronomic adjustment variable *(A)*.

**3.4 Yield Prediction and Quality Grading Algorithms**  
The final algorithmic grading relies on the mathematical fusion of the edge-derived visual features and the cloud-computed agronomic variables. First, the mean Visual Grade is calculated from the ResNet-50 output array:

Visual Grade = 1n i=1nVisual Featurei 

Simultaneously, the regression engine calculates the raw Agronomic Adjustment *(A)* and isolates negative field-management practices into an Agronomic Penalty *(P).* Crucially, to account for physiological yield losses from suboptimal harvest timing, a variety-specific maturity penalty wmaturityv  derived from the computer vision classification is integrated into the penalty calculation. This ensures that yield deductions caused by sucrose inversion in over-mature cane or excessive stalk moisture in not mature cane are mathematically enforced:

A = kwk ×xk  
P = k(0,wk ×xk) + wmaturityv

To prevent the computation of negative theoretical yields, the penalty is converted into a bounded Agronomic Multiplier (M): 

M \=0, 1 \+ P 

The final predictive yield outputs (y) are computed by scaling the official SRA Baselines (*B)* using the multiplier *(M).* These functions strictly bound the predictions, ensuring they do not exceed the theoretical maximums for the selected variety *(v),* ratoon stage *(s),* and farm size in hectares *(h)*:

yLKGTC=max0, minBLKGTCv,s, BLKGTCv,sM   
yTCHA= max0, minBTCHAv,sh, BTCHAv,sh ×M

Ultimately, the system outputs the final Predicted Quality Grade by directly fusing the AI visual assessment with the regression-based agronomic adjustment:

Predicted Quality Grade \= Visual Grade \+ A   
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
This section should provide enough detail to allow full replication of the study by suitably skilled investigators. Protocols for new methods should be included, but well-established protocols may simply be referenced.

**Table 2** Datasets

| Sugarcane Varieties and Maturity | Sample Images | Training Images | Testing Images | Validation Images | Total Images |
| ----- | ----- | ----- | ----- | ----- | ----- |
|  VMC 84-524 (Not Mature) |   |  |  |  |  1,000 |
|  VMC 84-524 (Mature) |   |  |  |  |  1,000 |
|  VMC 84-524 (Over Mature)  |   |  |  |  |  1,000 |
|  VMC 84-947 (Not Mature)  |  |  |  |  |  1,000 |
|  VMC 84-947 (Mature)  |   |  |  |  |  1,000 |
| VMC 84-947 (Over Mature)  |   |  |  |  |  |
|  |   |  |  |  |  |
|  |   |  |  |  |  |

**3.8 Computer Vision Pipeline (Plan, Explore, Build, Evaluate)**  
The development of the artificial intelligence imaging module adhered to the Plan, Explore, Build, and Evaluate (PEBE) project management framework which maintains a structured approach to modeling within an uncontrolled agricultural environments. The pipeline executes entirely on the mobile edge device to eliminate  latency and bandwidth costs of transmitting high-resolution images to the cloud.

**3.8.1 Image Pre-processing and Color Space Conversion (Plan and Explore)**  
Field-based agricultural imaging is inherently uncontrolled, with datasets affected by ambient illumination, canopy shadows, lens glare, and atmospheric dust. To address these anomalies during the Plan and Explore phases, the pipeline implements an automated pre-processing algorithm using OpenCV to convert standard RGB images into the HSV (Hue, Saturation, Value) color space. This transformation decouples lighting intensity (Value) from pigment (Hue) and saturation, which secures sucrose-related chromatic transitions from green to yellow-brown remain consistent regardless of diurnal timing or weather conditions.  
In a study of Ling et al. (2025), it has been discovered that robust image enhancement systems integrating Non-Local Means (NLM) and CLAHE filtering significantly improved plant leaf image quality, thereby enhancing detection accuracy. This validates the importance of pre-processing in agricultural pipelines, where lighting normalization directly mitigates false maturity readings.

**3.8.2 Sugarcane Localization via YOLOv8 (Build)**  
Sugarcane fields present noisy visual backgrounds cluttered with overlapping leaves, soil, and weeds. To ensure accurate maturity grading, the system isolates stalks using YOLOv8, chosen for its high inference speed and lightweight computational footprint optimized for mobile hardware. The model scans HSV-normalized images, generating bounding boxes to delineate the Region of Interest (ROI) and crop out irrelevant background noise, ensuring that subsequent classifiers evaluate only the cane rind.  
In a study of Gupta et al. (2025), it was discovered that YOLOv8 achieves a strong balance between accuracy and speed, which makes it ideal for real-time agricultural monitoring. For instance, a specialized “Sugarcane-YOLO” variant achieved a mean Average Precision (mAP@50) of 99.05% in seed sprout identification, while the RepNCSP-ELAN-YOLOv8n model reached 90.1% accuracy with only 4.6 MB model size and 14.33 ms latency. These results confirm YOLOv8’s suitability for edge deployment in sugarcane maturity analysis.

**3.8.3 Deep Feature Extraction via ResNet-50 (Evaluate)**  
Once stalks are localized, cropped ROIs are passed to a ResNet-50 CNN for deep feature extraction. Its residual learning architecture circumvents the vanishing gradient problem, which permits the network to capture granular morphological features such as internode spacing, rind texture anomalies, and localized color distributions. Unlike traditional binary classifiers, the terminal dense layers of ResNet-50 output a continuous 5-dimensional numerical array (Visual Featurei), representing the stalk’s maturity and visual grade.  
In a study by Desai and Ganatra (2025), it has been discovered that ResNet-50 consistently outperforms conventional architectures such as VGG-16 and AlexNet for plant-related feature extraction, as it achieves accuracies above 97% in tomato leaf disease classification. Similarly, Korade et al. (2025) reported that phenotype-based deep learning models using ResNet-50 achieved 97.14% accuracy in agricultural classification tasks. These findings validate ResNet-50’s role as a robust backbone for extracting maturity-related features in sugarcane.

**3.8.4 Combination of YOLOv8 and ResNet-50**  
The pipeline’s strength centers on the combination of YOLOv8 and ResNet-50. YOLOv8 rapidly isolates stalks with high localization precision, while ResNet-50 extracts deep morphological features for maturity grading. Together, they form a two-stage pipeline that balances speed and accuracy.  
In a study of Korade et al. (2025), it has been discovered that hybrid pipelines combining YOLO for region-of-interest extraction and ResNet for refinement significantly enhance predictive performance, achieving classification accuracies above 97% and F1-scores as high as 0.988. Similarly, Desai and Ganatra (2025) reported that a ResNet50-YOLOv8 hybrid model for cotton disease identification reached 98% accuracy, which presents the superiority of this fusion in agricultural contexts. These results confirm that the integration of YOLOv8 and ResNet-50 improves the feature set, reduces background noise, and enables precise maturity grading of sugarcane.

**3.8.5 Data Fusion and Yield Prediction**  
The extracted feature array is securely transmitted to the Flask backend, where it is fused with agronomic parameters, including irrigation schedules, soil health metrics, and local weather conditions. Regression-based paradigms like Gradient Boosting Decision Trees (GBDT) and XGBoost are employed to integrate structured tabular data with phenotypic features.  
In a study of Bienvenu et al. (2025), it was discovered that integrating digital phenotypic data into decision support systems improved multi-trait prediction tasks, which achieved an accuracy of 0.73 in sugarcane breeding contexts. This validates that combining visual maturity features with environmental and management data enhances predictive reliability, thereby securing scientifically valid yield forecasts.

**4\.  Results**  
	The results section should provide details of all of the experiments that are required to support the conclusions of the paper. There is no specific word limit for this section.  The section may be divided into subsections, each with a concise subheading.  The results section should be written in the past tense.   
Tables and captions must be centered. They should be produced in a spreadsheet program such as Microsoft Excel or in Microsoft Word. Type all text in tables using 9-point font on single line spacing. Type the caption above the table to the same width as the table.    
Tables should be numbered consecutively. Footnotes to tables should be typed below the table and should be referred to by superscript numbers. Tables should not duplicate results presented elsewhere in the manuscript (e.g., in graphs)

**Table 1** Table caption

| C1 | C2 | C3 | C4 |
| :---: | ----- | ----- | ----- |
| R1 |  |  |  |
| R2 |  |  |  |
| R3 |  |  |  |
| R4 |  |  |  |
| R5 |  |  |  |

If figures are inserted into the main text, type figure captions below the figure.  In addition, submit each figure individually as a separate file. Figures should be provided in a file format and resolution suitable for reproduction, e.g., EPS, JPEG or TIFF formats, without retouching. Photographs, charts and diagrams should be referred to as "Figure(s)" and should be numbered consecutively in the order to which they are referred

**Figure 1**  Figure caption

**Table 2** Table caption

|  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

**Figure 2**  Figure caption  
**5\.  Discussion**  
The discussion should spell out the major conclusions of the work along with some explanation or speculation about the significance of these conclusions. How do the conclusions affect the existing assumptions and models in the field? How can future research build on these observations? What are the key experiments that must be done?  The discussion should be concise and tightly argued. Conclusions firmly established by the presented data, hypotheses supported by the presented data, and speculations suggested by the presented data should be clearly identified as such. The results and discussion may be combined into one section, if desired.

**6\.  Conclusion**  
The Conclusion section restates the major findings and suggests further research.

**7\.  Acknowledgements (if any)**  
People who contributed to the work but do not fit criteria for authorship should be listed in the Acknowledgments, along with their contributions. It is the authors’ responsibility to ensure that anyone named in the acknowledgments agrees to being so named. The funding sources that have supported the work should be included in the acknowledgments.

**8\.  References**  
Briones, R., Galang, I. M., & Latigar, J. (2023). Transforming Philippine agri-food systems with digital technology: Extent, prospects, and inclusiveness. *Philippine Institute for Development Studies.* [https://pidswebs.pids.gov.ph/CDN/document/pidsdps2329.pdf](https://pidswebs.pids.gov.ph/CDN/document/pidsdps2329.pdf)  
Diyanah, S. M., Kozono, M., & Hazmi, A. (2024, December 29). Accelerating the digitalisation of the agriculture and food system in the ASEAN region. *Eria.org.* [https://www.eria.org/research/accelerating-the-digitalisation-of-the-agriculture-and-food-system-in-the-asean-region](https://www.eria.org/research/accelerating-the-digitalisation-of-the-agriculture-and-food-system-in-the-asean-region)  
Food and Agriculture Organization of the United Nations. (2025). Sugar cane production. *Our World in Data.* [https://ourworldindata.org/grapher/sugar-cane-production](https://ourworldindata.org/grapher/sugar-cane-production). Retrieved November 11, 2025\.  
Montesclaros, J. M. L., & Teng, P. S. (2023). Digital technology adoption and potential in Southeast Asian agriculture. *Asian Journal of Agriculture and Development, 20*(2), 7–30. [https://doi.org/10.37801/ajad2023.20.2.2](https://doi.org/10.37801/ajad2023.20.2.2)  
Oñal, P., Juntilla, R., & Ortega, J. (2021). Degree of farming innovations and the level of productivity of sugarcane farmers in the Visayas, Philippines. *European Journal of Agricultural and Rural Education (EJARE),* 57–67. [https://media.neliti.com/media/publications/382222-degree-of-farming-innovations-and-the-le-3875268d.pdf](https://media.neliti.com/media/publications/382222-degree-of-farming-innovations-and-the-le-3875268d.pdf)  
Pantoja, B., Alvarez, J., & Sanchez, F. (2019). Implementing sugarcane block farming for increased farm income and productivity. *Philippine Institute for Development Studies.* [https://pidswebs.pids.gov.ph/CDN/PUBLICATIONS/pidsrp0901.pdf](https://pidswebs.pids.gov.ph/CDN/PUBLICATIONS/pidsrp0901.pdf)  
Rose, D. C., & Chilvers, J. (2018). Agriculture 4.0: Broadening responsible innovation in an era of smart farming. *Frontiers in Sustainable Food Systems, 2*(87). [https://doi.org/10.3389/fsufs.2018.00087](https://doi.org/10.3389/fsufs.2018.00087)  
Solomon, S., Khumla, N., Manimekalai, R., & Misra, V. (2024). Prospects of diversification for sustainable sugar bioenergy industries in ASEAN countries. *Sugar Tech, 26,* 951–971. [https://link.springer.com/article/10.1007/s12355-024-01432-x](https://link.springer.com/article/10.1007/s12355-024-01432-x)  
Sugar Regulatory Administration (SRA). (2019). Overview of the sugarcane industry. [https://sra.gov.ph/storage/OVERVIEW-OF-THE-SUGARCANE-INDUSTRY\_September-2019.pdf](https://sra.gov.ph/storage/OVERVIEW-OF-THE-SUGARCANE-INDUSTRY_September-2019.pdf). Retrieved October 27, 2025\.

**9\. Appendices**

**Appendix A**   
**Mobile Application (User Interface)**

**Web Application (Admin and Super Admin Side)** 

**Appendix B**  
**Written Communication**

**Appendix C**   
**Data Gathering Process**

**Appendix D**  
**Certificate of Research Ethics Approval**

**10\. Author(s) Biodata**   
Faye N. Cuevas is an undergraduate Bachelor of Science in Computer Science student at STI West Negros University. Her academic focus includes artificial intelligence, predictive analytics, and computer vision. She applies deep learning for image analysis, with practical experience in Python programming and project management, contributing technical expertise to research initiatives.  
Windy D. Elipse is an undergraduate Bachelor of Science in Computer Science at STI West Negros University. She has expertise in database management using PostgreSQL, complemented by practical knowledge in Python programming and UI/UX design. Her contributions emphasize integrating data-driven architectures with user-centered design.   
John Rolo C. Flores is an undergraduate BS Computer Science student at STI West Negros University. His academic pursuits emphasize programming and artificial intelligence, with strong proficiency in Python and advanced expertise in UI/UX design. He contributes to projects that integrate functional programming with user-centered interfaces.   
Lynol I. Ibarra is a faculty member at STI West Negros University. His professional expertise spans artificial intelligence and cybersecurity, contributing to both academic instruction and applied research. With extensive knowledge in these domains, he supports innovative computational solutions while guiding students in integrating secure, intelligent systems that advance institutional research and technological development.  
‘

