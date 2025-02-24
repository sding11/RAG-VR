This repository code implements the functionality of the RAG-VR system.
# Outline
<p align="center">
  <img src="https://raw.githubusercontent.com/sding11/RAG-VR/refs/heads/main/User.png" width="25%">
  <img src="https://raw.githubusercontent.com/sding11/RAG-VR/refs/heads/main/Server.png" width="25%">
  <img src="https://raw.githubusercontent.com/sding11/RAG-VR/refs/heads/main/UI.png" width="40%">
</p>

## Testing Setup
* VR device: Meta Quest 3
* Edge server GPU: NVIDIA RTX 2000 Ada
* Unity editor version: 2022.3.52f1
* LLM:llama3.1:8b

## Implementation of Dataset-based Evaluation
### Training
a. Download the necessary packages
```
pip install langchain langchain_experimental langchain-groq --use-deprecated=legacy-resolver
pip install -U langchain-ollama
pip install faiss-cpu
pip install transformers
pip install pandas --upgrade
pip install openpyxl --upgrade
pip install scikit-learn
```
b. Open a terminal in "RAG-VR" folder, Run:
```
./dataset-based evaluation/train.py
```

### Testing
c. Download Ollama from https://ollama.com/. Then run the following commands to download the LLM (llama3.1:8b).
```
ollama run llama3.1:8b
```

d. Open a terminal in "RAG-VR" folder, Run:
```
./dataset-based evaluation/test.py
```

## Implementation of Real-world Testing
### Unity Setup
a. In Unity Hub, create a new 3D Unity project. 

b. Navigate to Window>Asset Store.  Search for the virtual reality game (e.g., the 'Viking Village URP' game.) in the Asset Store, and select 'Buy Now' and 'Import'.

c. In the Hierarchy Window, select the game objects that may be queried. In the Inspector view on the right, click "Add Tag" in the drop-down option of the "Tag" box and click "+" to create the Tag. Then, give each selected object one of these new tags.

d. Create the "Scripts" folder under the project's "Assets" folder. Save the GetMainCameraInfo.cs, GetObjectInformation.cs, PlayerController.cs, and SpeechRecognitionTest.cs files to the "Scripts" folder. In the "SpeechRecognitionTest.cs", modify the "serverUrl" to the current IPv4 address.

e. Create four new GameObjects in the Hierarchy Window and bind the imported C# scripts to each GameObject.

f. In the Hierarchy Window, right-click, select "UI", and create 2 "Text-TextMeshPro", and 2 "Button-TextMeshPro", named "InputText", "OutText", "StartButton" and "StopButton" respectively. Adjust their position for proper observation. And add prompt text for them to display.

g. Select your connected target device (Meta Quest 3) and click 'Build and Run'.

h. The output APK package will be saved to the file path you specify, while the app will be installed on the Meta Quest 3 device connected to your computer.

### Edge Server Setup

i. Load the code in the "server" folder on the local server. Select the system you want to run in server.py. The "rag_vr_process_question" function calls the RAG-VR system, "in_context_process_question" calls the in-context LLM system, and "vanilla_rag_process_question" calls the Vanilla-RAG system.

j. Disconnect the Meta Quest 3 from the computer. After setting up a new Guardian Boundary, the virtual reality game with RAG-VR will be automatically loaded.

k. Click the button on the UI interface according to the prompt to start and end the query.
