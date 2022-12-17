Note:
-----------------------------------------------------------------------------------------------------------------------------
- Machine learning models
	- KNeighborsClassifier
	- PrincipalComponentAnalysis
- Deep learning models
	- DeepLearningModel (TabNet Classifier)
	- T-Distributed Stochastic Neighbor Embedding
- HeartRate Features Generation
	- RMSSD
	- SDNN
	- PoincareDiagrammFromRRIntervals
- Cognitive Load Estimation
	- CognitiveLoadEstimationEEG
	- CognitiveLoadEstimationHeartRateAndSkin
	- CognitiveLoadEstimationPupilometry
- Feature selection
	- FeatureSelection
	- CorrelationMatrixSklearn
	- BruteForceParameterOptimization_DL (DeepLearningModel)
	- BruteForceParameterOptimization_KNM (MachineLearningModel)
- Plots
	- BehaviorValidityScoreAbstraction
	- CreateSimulationSicknessPlots
	- CreateUvqPlotsAndResults
- Converter
	- ConvertGoogleFormsFormatToMScvsFormat
	- ConvertBitalinoRawDataForBioSPPy
- Stresstest with Kaiser rule
	- PrincipalComponentAnalysis

Machine learning models (KNeighborsClassifier):
-----------------------------------------------------------------------------------------------------------------------------
- Used for classify the given fusion dataset from the AutomationBuild script of R-Project
- Dataset file path: "./All_Participents_Clusterd_WaveSum_DataFrame.csv"

Deep learning models (TabNet Classifier):
-----------------------------------------------------------------------------------------------------------------------------
- Used for classify the given fusion dataset from the AutomationBuild script of R-Project
- Dataset for classification file path: "./All_Participents_Clusterd_WaveSum_DataFrame.csv"
- Dataset for testing (ecg, eda, eeg, eye)	sensor data file path: "./All_Participents_Clusterd_WaveSum_DataFrame_01.csv"
- Dataset for testing (ecg, eda, eeg) 		sensor data file path: "./All_Participents_Clusterd_WaveSum_DataFrame_02.csv"
- Dataset for testing (ecg, eda) 			sensor data file path: "./All_Participents_Clusterd_WaveSum_DataFrame_03.csv"
- Dataset for testing (ecg) 				sensor data file path: "./All_Participents_Clusterd_WaveSum_DataFrame_04.csv"
- After the training the model is saved under: "./output//iterration_{iterration_number}/trained_tabnet_model.zip"

HeartReate Features Generation:
-----------------------------------------------------------------------------------------------------------------------------
- Used for gerating time domain and none parametric HeartRate measurement feature
- Path: "../BitalinoTools/Tools/HRV-TimeDomain/*.*"
- For using read the readme in the given path.

Cognitive Load Estimation:
-----------------------------------------------------------------------------------------------------------------------------
- Used for cognitive load estimating in the given fusion dataset from the AutomationBuild script of R-Project
- Dataset stage 0 file path: "./All_Participents_Stage0_DataFrame.csv"
- Dataset stage 1 file path: "./All_Participents_Stage1_DataFrame.csv"

Feature selection:
-----------------------------------------------------------------------------------------------------------------------------
- Created tests scripts for choosing the best quilfied features for classification problem
- Dataset file path: "./All_Participents_Clusterd_WaveSum_DataFrame.csv"
- Dataset file path: "./All_Participents_Condition-C_WaveSum_DataFrame.csv"

Plots (BehaviorValidityScoreAbstraction):
-----------------------------------------------------------------------------------------------------------------------------
-

Plots (CreateSimulationSicknessPlots):
-----------------------------------------------------------------------------------------------------------------------------
-

Plots (CreateUvqPlotsAndResults):
-----------------------------------------------------------------------------------------------------------------------------
- Used for processing with ANOVA method to generate a compare to original experiment and to create results
- Creates plots like the original experiment
- Dataset file path = "../RTools/{selected condition}/RResults/Questionnaires/AllUncannyValleyConditionStatisticResults_DataFrame.csv"

Converter (ConvertGoogleFormsFormatToMScvsFormat):
-----------------------------------------------------------------------------------------------------------------------------
- 

Converter (ConvertBitalinoRawDataForBioSPPy):
-----------------------------------------------------------------------------------------------------------------------------
- 

Stresstest with Kaiser rule
-----------------------------------------------------------------------------------------------------------------------------
- Used for selecting the best number of components for the principal component analysis method with the Kaiser rule
- Dataset file path = "./All_Participents_Clusterd_WaveSum_DataFrame.csv"
