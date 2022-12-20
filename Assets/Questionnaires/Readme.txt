
Using original Questionnaire Toolkit
------------------------------------------------------------------------------------------------------------------
@inproceedings{feick2020vrqt,
author = {Feick, Martin and Kleer, Niko and Tang, Anthony and Kr\"{u}ger, Antonio},
title = {The Virtual Reality Questionnaire Toolkit},
year = {2020},
isbn = {9781450375153},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3379350.3416188},
doi = {10.1145/3379350.3416188},
location = {Virtual Event, USA},
series = {UIST '20 Adjunct}
}
------------------------------------------------------------------------------------------------------------------
Copy the content of current version of Questionnaires Asset folder in this (./) folder:
https://github.com/MartinFk/VRQuestionnaireToolkit/tree/master/Questionnaires/Questionnaire/Assets/Questionnaires

In addition:
- copy the content of the Resource Asset folder in the ../Resources folder:
https://github.com/MartinFk/VRQuestionnaireToolkit/tree/master/Questionnaires/Questionnaire/Assets/Resources
- remove the content of the Plugin-Folder (./Plugin/*), to ensure that there are not Plugin conflicts.


Using VRSTK integrated Questionnaire Toolkit implemention
------------------------------------------------------------------------------------------------------------------
To use this toolkit with VRSTK features, the VRSTK-Questionnaire-Unity package must be installed.

Therefore, additional steps must be performed in the original toolkit package:
- remove the content of the Prefab-Folder (./Prefab/*)
- remove the content of the Scripts-Folder (./Scripts/*)
- unzip the Questionnaires.zip file into current folder (./*)

