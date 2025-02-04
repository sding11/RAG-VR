import os
from transformers import DistilBertTokenizer
import torch
import pandas as pd
import json
from langchain_ollama import OllamaLLM
from torch.nn.functional import normalize

# Load the dual-tower model
def load_dual_tower_model(model_path, device):
    model = DualTowerModel()  # Use the new version of the dual-tower model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# Generate embeddings using the model
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
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif mode == "doc":
            outputs = model.doc_encoder(input_ids=input_ids, attention_mask=attention_masks)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            raise ValueError("Mode must be 'query' or 'doc'.")

    return normalize(embeddings, p=2, dim=1)  # L2 normalization

# Similarity calculation function
def compute_similarity(question_embeddings, object_embeddings):
    return torch.cosine_similarity(question_embeddings, object_embeddings, dim=-1)

# Retrieve top-k most relevant documents using the dual-tower model
def retrieve_with_dual_tower(model, tokenizer, question, documents, device, k=6):
    question_embeddings = get_embeddings(model, [question], tokenizer, device, mode="query")

    object_retrieved_index = documents["Retrieved Index"].astype(str).tolist()

    object_embeddings = get_embeddings(model, object_retrieved_index, tokenizer, device, mode="doc")

    # Compute cosine similarity
    similarities = torch.mm(question_embeddings, object_embeddings.T).squeeze(0)

    top_k_indices = torch.topk(similarities, k=k).indices.cpu().tolist()

    object_retrieved_list = [object_retrieved_index[idx] for idx in top_k_indices]

    top_k_rows = documents.iloc[top_k_indices]

    return top_k_rows, object_retrieved_list, similarities.cpu().tolist()

# Use LLM to generate answers
def answer_with_llm(question, context, llm):
    """
    Use LLM to generate an answer based on the retrieved context.
    """
    prompt = f"""
    The following is relevant context extracted from the database, 'Type' means type of the object and 'Name' means the name of the object. There are as many objects as there are names:

    {context}

    Question: {question}

    Answer briefly and concisely based on the provided context, don't explain.
    """
    response = llm.invoke(prompt)
    return response

# Save the results of the question-answering process
def save_rag_results(output_dir, scene_name, results):
    """
    Save the questions and their corresponding answers and retrieval results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}_rag_results.csv")

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Results saved to {output_path}")

# RAG pipeline
def rag_pipeline(question, documents, dual_tower_model, tokenizer, llm, device, k=6):
    """
    RAG pipeline: Dual-tower retrieval + LLM generation
    """
    # Retrieve the top-k relevant documents
    top_k_rows, top_k_indices, similarities = retrieve_with_dual_tower(
        dual_tower_model, tokenizer, question, documents, device, k
    )

    # Concatenate retrieved rows into context
    context = "\n".join(top_k_rows.apply(lambda row: row.to_json(), axis=1))

    # Generate an answer using LLM
    answer = answer_with_llm(question, context, llm)

    return question, answer, top_k_indices, similarities

# Example execution
if __name__ == "__main__":
    # Initialize LLM
    llm = OllamaLLM(model="llama3.1:8b")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    output_dir = "test_result/rag_vr/single"

    # Assuming `documents` is a DataFrame containing Name, Type, etc.
    for j in range(1):
        i = 
        documents = all_scene_dataframes[i]
        test_set = test_set_dataframes[i]
        questions = test_set["Questions"].tolist()
        standard_answers = test_set["Standard answers"].tolist()
        standard_retrieved_name = test_set["Retrieved Name"].tolist()
    
        scene_name = all_scene_dataframes[i]["Scene"][0]
    
        dual_tower_model_path = f"C:/Users/sding1/aaa_AR+LLM/RAG_VR/train/{scene_name}/{scene_name}_dual_tower_model.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dual_tower_model = load_dual_tower_model(dual_tower_model_path, device)
    
        # Store results for each question
        results = {
            "Question": [],
            "Answer": [],
            "Retrieved Object Name": [],
            "Similarities": [],
            "Standard Answer": [],  # Add standard answer
            "Standard Retrieved Name": []  # Add standard retrieved object name
        }
    
        # Iterate through questions, standard answers, and standard retrieved names
        index_code = 0
        for question, std_answer, std_retrieved_name in zip(questions, standard_answers, standard_retrieved_name):
            # Execute RAG pipeline
            question, answer, top_k_indices, similarities = rag_pipeline(
                question, documents, dual_tower_model, tokenizer, llm, device, k=10
            )
            print(f"{index_code}: Q:{question}--> A:{answer}")
            # Store results
            results["Question"].append(question)
            results["Answer"].append(answer)
            results["Retrieved Object Name"].append(top_k_indices)
            results["Similarities"].append(json.dumps(similarities))
            results["Standard Answer"].append(std_answer)
            results["Standard Retrieved Name"].append(std_retrieved_name)
            index_code += 1
    
        # Save results for the current scene
        save_rag_results(output_dir, scene_name, results)
