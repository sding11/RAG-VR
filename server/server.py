from fastapi import FastAPI, Request
import uvicorn
import logging
import os
import time
from in_context import in_context_process_question
# from rag_vr import rag_vr_process_question  
from test import rag_vr_process_question  
from save_latency import update_csv_with_response_time  
from save_server_time import update_csv_with_server_time
from vanilla_rag import vanilla_rag_process_question

# set log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state = {"server_time": None}


app = FastAPI()

# answer list
answers = {
    " hello, server": "Hello, client! How can I assist you?",
    " how are you": "I'm just a server, but thank you for asking!",
    " goodbye.": "Goodbye! Hope to assist you again soon."
}



@app.post("/pose")
async def post_pose(request: Request):
    """
    Receive the question and pose data in JSON format and process them separately.
    """
    server_start_time = time.time()
    global player_json

    # Parse the incoming JSON data.
    post_data = await request.json()

    # Extract the user input text and camera data.
    question = post_data.get("text", "").lower()
    position = post_data.get("position", {})
    rotation = post_data.get("rotation", {})

    # Save the pose data to 'pose_data'.
    player_json = {"position": position, "rotation": rotation}
    
    print(player_json)

    # Pass the question and pose data to the RAG system for processing.
    answer = rag_vr_process_question(question, player_json)  # rag_vr
    # answer = in_context_process_question(question, player_json)  # in-context
    # answer = vanilla_rag_process_question(question, player_json)
    
    # Server end time.
    server_end_time = time.time()
    global server_time
    state["server_time"] = (server_end_time - server_start_time) * 1000  # Convert to milliseconds.

    return {"answer": answer}


@app.post("/time")
async def post_time(request: Request):
    """
    receive response time
    """
    post_data = await request.json()
    response_time = post_data.get("responseTime", None)
    # change to ms
    response_time = response_time * 1000
    
    # get server time
    server_time = state.get("server_time", None)
    
    # save latencytime
    save_latency_file_path = r"server/test_result/rag_vr/office_with_conference_room_rag_results.csv"
    update_csv_with_response_time(server_time, response_time, save_latency_file_path)

    if response_time is not None:
        logger.info(f"Received response time: {response_time} ms")
        return {"message": "Response time logged successfully"}
    else:
        logger.warning("Invalid data received for response time")
        return {"error": "Invalid data received"}
    
    

if __name__ == "__main__":
    # Run the server using Uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    