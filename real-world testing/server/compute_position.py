import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def update_documents_with_player_info(player_json, documents):

    player_position = np.array([
        player_json['position']['x'],
        player_json['position']['y'],
        player_json['position']['z']
    ])
    player_rotation = np.array([
        player_json['rotation']['x'],
        player_json['rotation']['y'],
        player_json['rotation']['z'],
        player_json['rotation']['w']
    ])

    player_rotation_matrix = R.from_quat(player_rotation).as_matrix()

    distances = []
    directions = []

    for _, row in documents.iterrows():
        object_position = parse_position(row['Position'])

        relative_position = compute_relative_position(object_position, player_position, player_rotation_matrix)

        distance = np.linalg.norm(relative_position)
        direction = format_direction(relative_position, row['Name'])

        distances.append(round(distance, 2))  
        directions.append(direction)

    documents['Distance to Player'] = distances
    documents['Direction from Player'] = directions


def parse_position(position):

    if isinstance(position, str):
        position = eval(position)  
    return np.array([position['x'], position['y'], position['z']])


def compute_relative_position(object_position, player_position, player_rotation_matrix):

    relative_position = np.linalg.inv(player_rotation_matrix).dot(object_position - player_position)
    return relative_position


def format_direction(relative_position, object_name):

    x, y, z = relative_position
    direction = f"{object_name.capitalize()} is" 

    if z > 0:
        direction += " front"
    elif z < 0:
        direction += " back"
    
    if x > 0:
        direction += " right"
    elif x < 0:
        direction += " left"

    direction += " of the player."
    return direction
