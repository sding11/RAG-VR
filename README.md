This repository code implements the functionality of the RAG-VR system. **The full code will be published after the paper is accepted.**
# Outline
![](https://raw.githubusercontent.com/sding11/RAG-VR/refs/heads/main/UI.png)

## Testing Setup
* VR device: Meta Quest 3
* Edge server: Dell Precision 3591 laptop, Intel® Core™ Ultra 7 165H 1.40 GHz CPU,  NVIDIA RTX 2000 Ada GPU
* Unity editor version: 2022.3.52f1

## Implementation of RAG-VR
a. In Unity Hub, create a new 3D Unity project. 

b. Navigate to Window>Asset Store.  Search for the virtual reality game (e.g., the 'Viking Village URP' game.) in the Asset Store, and select 'Buy Now' and 'Import'.

c. In the Hierarchy Window, select the game objects that may be queried. In the Inspector view on the right, click "Add Tag" in the drop-down option of the "Tag" box and click "+" to create the Tag. Then, give each selected object one of these new tags.

d. Create the "Scripts" folder under the project's "Assets" folder. Save the GetMainCameraInfo.cs, GetObjectInformation.cs, PlayerController.cs, and SpeechRecognitionTest.cs files to the "Scripts" folder

e. Create four new GameObjects in the Hierarchy Window and bind the imported C# scripts to each GameObject.

f. In the Hierarchy Window, right-click, select "UI", and create 2 "Text-TextMeshPro", and 2 "Button-TextMeshPro", named "InputText", "OutText", "StartButton" and "StopButton" respectively. Adjust their position for proper observation. And add prompt text for them to display.

g. Select your connected target device (Meta Quest 3) and click 'Build and Run'.

h. The output APK package will be saved to the file path you specify, while the app will be installed on the Meta Quest 3 device connected to your computer.

i. Disconnect the Meta Quest 3 from the computer. After setting up a new Guardian Boundary, the virtual reality game with RAG-VR will be automatically loaded.

