import os
import torch
import pandas as pd
import json
from langchain_ollama import OllamaLLM
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_scheduler
from compute_position import update_documents_with_player_info
import time
from cloud_fireworks import get_fireworks_answer


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



class DualTowerModel(nn.Module):
    def __init__(self):
        super(DualTowerModel, self).__init__()
        self.query_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.doc_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(self.query_encoder.config.hidden_size, 128) 

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        query_hidden = self.query_encoder(query_input_ids, attention_mask=query_attention_mask)[0][:, 0, :]
        query_embeds = F.normalize(self.fc(query_hidden), p=2, dim=1)

        doc_hidden = self.doc_encoder(doc_input_ids, attention_mask=doc_attention_mask)[0][:, 0, :]
        doc_embeds = F.normalize(self.fc(doc_hidden), p=2, dim=1)

        return query_embeds, doc_embeds

def load_dual_tower_model(model_path, device):
    model = DualTowerModel()  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def get_embeddings(model, texts, tokenizer, device, mode="query"):
    model.eval()
    input_ids, attention_masks = [], []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    with torch.no_grad():
        if mode == "query":
            outputs = model.query_encoder(input_ids=input_ids, attention_mask=attention_masks)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
        elif mode == "doc":
            outputs = model.doc_encoder(input_ids=input_ids, attention_mask=attention_masks)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
        else:
            raise ValueError("Mode must be 'query' or 'doc'.")

    return normalize(embeddings, p=2, dim=1)  





def compute_similarity(question_embeddings, object_embeddings):
    return torch.cosine_similarity(question_embeddings, object_embeddings, dim=-1)


def retrieve_with_dual_tower(model, tokenizer, question, documents, device, k=6):
    question_embeddings = get_embeddings(model, [question], tokenizer, device, mode="query")

    object_retrieved_index = documents["Retrieved Index"].astype(str).tolist()

    object_embeddings = get_embeddings(model, object_retrieved_index, tokenizer, device, mode="doc")


    similarities = torch.mm(question_embeddings, object_embeddings.T).squeeze(0)  


    top_k_indices = torch.topk(similarities, k=k).indices.cpu().tolist()
    
    object_retrieved_list = [object_retrieved_index[idx] for idx in top_k_indices]

    top_k_rows = documents.iloc[top_k_indices]

    return top_k_rows, object_retrieved_list, similarities.cpu().tolist()



# Use the LLM to answer the question
def answer_with_llm(question, context, llm):
    prompt = f"""
    The following is relevant context extracted from the database, 'Type' means type of the object and 'Name' means means name of the object, there are as many objects as there are names:

    {context}

    Question: {question}

    Answer briefly and concisely based on the provided context, don't explain.
    """
    response = llm.invoke(prompt)
    return response



# save result
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



def rag_pipeline(question, documents, dual_tower_model, tokenizer, llm, device, k=6):
    top_k_rows, top_k_indices, similarities = retrieve_with_dual_tower(
        dual_tower_model, tokenizer, question, documents, device, k
    )

    context = "\n".join(top_k_rows.apply(lambda row: row.to_json(), axis=1))

    answer = answer_with_llm(question, context, llm)

    return question, answer, top_k_indices, similarities



def rag_vr_process_question(question, player_json):
    
    current_dir = os.getcwd()

    input_scene_dir = os.path.join(current_dir, 'scene_information_csv')

    all_scene_dataframes, all_scene_file_paths = read_all_scene_info_csv_files(input_scene_dir)
    

    

    llm = OllamaLLM(model="llama3.1:8b")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    
    documents = all_scene_dataframes[0]
        
    update_documents_with_player_info(player_json, documents)
    
    dual_tower_model_path = f"server/office_with_conference_room_dual_tower_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_tower_model = load_dual_tower_model(dual_tower_model_path, device)

    results = {
            "Question": [],
            "Answer": [],
            "Retrieved Object Name": [],
            "Similarities": [],
            "Computation Latency (ms)": []
    }
    

    start_time = time.time()
    question, answer, top_k_indices, similarities = rag_pipeline(
                question, documents, dual_tower_model, tokenizer, llm, device, k=6
            )
    end_time = time.time()
    computation_latency = (end_time - start_time) * 1000  
    
    results["Question"].append(question)
    results["Answer"].append(answer)
    results["Retrieved Object Name"].append(top_k_indices)
    results["Similarities"].append(json.dumps(similarities))
    results["Computation Latency (ms)"].append(computation_latency)
    
    print(f"Q:{question}--> A:{answer}")
    print(f"Computation Latency: {computation_latency} ms")  

    output_dir = "test_result/rag_vr"
    scene_name = f"office_with_conference_room"
    save_rag_results(output_dir, scene_name, results)    
    
    return answer