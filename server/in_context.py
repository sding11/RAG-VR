import os
import pandas as pd
import json
from compute_position import update_documents_with_player_info
import time
from ollama import Client


def read_all_scene_info_csv_files(input_dir):


    dataframes = [] 
    file_names = []  

    for scene_folder in os.listdir(input_dir):
        scene_folder_path = os.path.join(input_dir, scene_folder)

        if os.path.isdir(scene_folder_path):
            for file_name in os.listdir(scene_folder_path):
                if file_name.endswith(f"{scene_folder}_input_info.csv"):  
                    file_path = os.path.join(scene_folder_path, file_name)
                    
                    try:
                        df = pd.read_csv(file_path)
                        if 'Index' in df.columns:
                            df = df.drop(columns=['Index'])  
                        dataframes.append(df)
                        file_names.append(file_path)
                        print(f"Successfully read: {file_path}")
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")

    return dataframes, file_names

def generate_in_context_llm_answer(documents, question):

    client = Client(host='http://localhost:11434')

    scene_content = documents.to_dict(orient='records')

    scene_content_json = json.dumps(scene_content, ensure_ascii=False)
    
    prompt = (
                f"This file describes what is stored in the scene. Each row contains information about an object.\n\n"
                f"{scene_content_json}\n\n"
                f"Please answer the following question briefly without explaining:\n{question}"
            )
    
    start_time = time.time()

    response = client.chat(model="llama3.1:8b", messages=[
        {
            "role": "user",
            "content": prompt
        },
    ])
    

    end_time = time.time()
    computation_latency = (end_time - start_time) * 1000  
    
    answer = response["message"]["content"].strip()

    return answer,computation_latency


def save_rag_results(output_dir, scene_name, results):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}_rag_results.csv")

    if not os.path.exists(output_path):
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Results created and saved to {output_path}")
    else:
        existing_df = pd.read_csv(output_path)
        new_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Results appended and saved to {output_path}")

def in_context_process_question(question, player_json):

    current_dir = os.getcwd()

    input_scene_dir = os.path.join(current_dir, 'scene_information_csv')


    all_scene_dataframes, all_scene_file_paths = read_all_scene_info_csv_files(input_scene_dir)

    documents = all_scene_dataframes[0]

    update_documents_with_player_info(player_json, documents)
    

    results = {
            "Question": [],
            "Answer": [],
            "Computation Latency (ms)": [] 
    }
    

    answer,computation_latency = generate_in_context_llm_answer(documents,question)
    
    
    
    results["Question"].append(question)
    results["Answer"].append(answer)

    results["Computation Latency (ms)"].append(computation_latency)
    
    print(f"Q:{question}--> A:{answer}")
    print(f"Computation Latency: {computation_latency} ms") 

    output_dir = "test_result/rag_vr"
    scene_name = f"office_with_conference_room"
    save_rag_results(output_dir, scene_name, results)    
    
    return answer