# Virtual Reality Scientific Toolkit (VRSTK)

The Virtual Reality Scientific Toolkit facilitates the creation and execution of experiments in VR environments by making object tracking and data collection easier.

This repo is provided under a MIT License.


## Features
- :world_map: Setup and Control
	- Structuring study phases and conditions
	- Live input and control for the operator with custom interfaces
- :movie_camera: Tracking of various elements
	- Participant movement
	- Gaze & Eye (with Vive Pro Eye) 
	- GameObjects
- :vhs: Scene Replay
- :file_folder: Import & Export of JSON data
- :bar_chart: Analysis templates with R
- :people_holding_hands: Multiplayer using [Mirror](https://github.com/vis2k/Mirror) (WiP)
	- Embodiment (WiP)
	- Asymmetric Normalization (WiP)

## Quickstart
Visit our wiki for quickstart instructions

[:de: Quickstart](https://github.com/ixperience-lab/VRSTK/wiki/Quickstart-German)\
[:uk: Quickstart](https://github.com/ixperience-lab/VRSTK/wiki/Quickstart-English)



## Requirements
- Unity3D 2020.3.x
- XR Unity Packages
	- OpenXR min. version 1.3.1
	- XR Plugin Management min. version 4.2.x
	- XR Interaction Toolkit min. version 2.2.0
- New Unity Input System min. version 1.3.0
#### Optional Requirements
- :eyes: [Eye-Tracking with Vive Pro Eye](https://developer-express.vive.com/resources/vive-sense/eye-and-facial-tracking-sdk/)
- :brain: [EEG with Emotiv](https://github.com/Emotiv/unity-plugin)


## Related Work

M. Wölfel, D. Hepperle, C. F. Purps, J. Deuchler and W. Hettmann, "Entering a new Dimension in Virtual Reality Research: An Overview of Existing Toolkits, their Features and Challenges," 2021 International Conference on Cyberworlds (CW), 2021, pp. 180-187, doi: 10.1109/CW52790.2021.00038.

Abstract: Virtual reality becomes a medium to be explored for itself, to study human factors and human behavior within these worlds, and to infer possible behavior in the real world. Among many advantages, building test routines in virtual environments remains a challenge due to the lack of established procedures and toolkits. To encourage research in this direction and lower the barrier to entry, it is necessary to simplify the process of setting up a research environment in virtual reality by providing appropriate toolkits. This paper discusses what challenges need to be overcome, what features might be relevant, and compares available toolkits.
URL: [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9599343&isnumber=9599338](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9599343&isnumber=9599338)

J. Deuchler, D. Hepperle and M. Wölfel, "Asymmetric Normalization in Social Virtual Reality Studies," 2022 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW), 2022, pp. 51-53, doi: 10.1109/VRW55335.2022.00019.

Abstract: We introduce the concept of asymmetric normalization, which refers to decoupling sensory self-perception from the perception of others in a shared virtual environment to present each user with a normalized version of the other users. This concept can be ap-plied to various avatar-related elements such as appearance, location, or non-verbal communication. For example, each participant in a polyadic virtual reality study can see other participants at an average height of the respective test population, while individual participants continue to see themselves embodied according to their actual height. We demonstrate in a pilot experiment how asymmetric normalization enables the acquisition of new information about social interactions and promises to reduce bias to promote replicability and external validity.
URL: [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9757601&isnumber=9757255](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9757601&isnumber=9757255)

