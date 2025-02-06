import time
import os
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from in_context import save_rag_results
from compute_position import update_documents_with_player_info

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def generate_original_rag_answer(scene_data, question):
    
    llm = OllamaLLM(model="llama3.1:8b")
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    
    documents = []
    for _, row in scene_data.iterrows():
        content = row.to_json() 
        documents.append(Document(page_content=content))

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_documents = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(split_documents, embeddings)
    

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    

    combine_docs_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "The following is relevant context extracted from the database, 'Type' means type of the object and 'Name' means means name of the object, there are as many objects as there are names: \n\n"
            "{context}\n\n"
            "Question: {input}\n"
            "Answer briefly and concisely based on the provided context, don't explain."
        ),
    )
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=combine_docs_prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)
    
    start_time = time.time()

    response = retrieval_chain.invoke({"input": question})

    end_time = time.time()
    computation_latency = (end_time - start_time) * 1000  
    
    answer = response["answer"].strip()
    
    return answer,computation_latency

# Call
def vanilla_rag_process_question(question, player_json):

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

    answer,computation_latency = generate_original_rag_answer(documents, question)
    
    
    
    results["Question"].append(question)
    results["Answer"].append(answer)

    results["Computation Latency (ms)"].append(computation_latency)
    
    print(f"Q:{question}--> A:{answer}")
    print(f"Computation Latency: {computation_latency} ms")  

    output_dir = "test_result/rag_vr"
    scene_name = f"office_with_conference_room"
    save_rag_results(output_dir, scene_name, results)    
    
    return answer