using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class GetMainCameraInfo : MonoBehaviour
{
    // define a class to store Camera information
    [System.Serializable]
    public class CameraInfo
    {
        public Vector3 position;
        public Quaternion rotation;
    }

    // Start is called before the first frame update
    void Start()
    {
        // require mainCamera information
        Camera mainCamera = Camera.main;

        if (mainCamera != null)
        {
            // create CameraInfo object
            CameraInfo cameraInfo = new CameraInfo
            {
                position = mainCamera.transform.position,
                rotation = mainCamera.transform.rotation
            };

            // change Camera information series to JSON
            string json = JsonUtility.ToJson(cameraInfo, true);

            // define save route
            string directoryPath = ".\\scene_information";
            string filePath = directoryPath + "\\player.json";



            // write JSON to documents
            File.WriteAllText(filePath, json);

            // output save route
            Debug.Log($"main CameraInfo has save to: {filePath}");
        }
        else
        {
            Debug.LogError("No mainCamera found!");
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}