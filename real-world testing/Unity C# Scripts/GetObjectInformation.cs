using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using JetBrains.Annotations;
using System.IO;
using System.Xml.Linq;
using Unity.VisualScripting;



public class GetObjectInformation : MonoBehaviour
{


    // Store all unique tags
    public List<string> Object_Tag_List = new List<string>();

    // return all unique tags list in the scene
    public List<string> FetchUniqueTagsInScene()
    {
        // Used to temporarily store tags to avoid duplicates
        HashSet<string> uniqueTags = new HashSet<string>();

        // search for all Game Objects
        GameObject[] allObjects = Resources.FindObjectsOfTypeAll<GameObject>();

        foreach (GameObject item in allObjects)
        {
            if (item.scene.isLoaded)
            {
                // Check whether the object has a Tag that is not empty or unmarked
                if (!string.IsNullOrEmpty(item.tag) && item.tag.StartsWith("custom_"))
                {
                    uniqueTags.Add(item.tag);
                }
            }
        }

        // Copy the unique Tag to List and return
        Object_Tag_List = new List<string>(uniqueTags);
        return Object_Tag_List;
    }

    [Serializable]
    public class Certain_Object
    {
        public string Tag;
        public string Name;

        // position: relative to world
        public Vector3 Position;
        // 
        public Quaternion Rotation;
        // localposition: relative to player
        public Vector3 LocalPosition;
        // 
        public Vector3 EulerAngles;


        // Indicates if the object is interactable
        public bool IsInteractable;

        public string ObjectMaterial_string;

        public string ObjectColor_string;
    }

    [Serializable]
    public class Scene_Objects
    {
        public Certain_Object[] scene_Objects_List;
    }

    // The maximum number of objects stored in a data set
    public const int MAX_OBJECTS = 1000;

    private Dictionary<Color, string> colorNames = new Dictionary<Color, string>
    {
        { Color.red, "Red" },
        { Color.green, "Green" },
        { Color.blue, "Blue" },
        { Color.yellow, "Yellow" },
        { Color.white, "White" },
        { Color.black, "Black" },
        { new Color(1.0f, 0.65f, 0.0f, 1.0f), "Orange" },
        { new Color(0.5f, 0.0f, 0.5f, 1.0f), "Purple" }
    };

    public string GetColorName(Color objectColor)
    {
        // Find the closest color
        float minDistance = float.MaxValue;
        string closestColorName = "Unknown";

        foreach (var colorPair in colorNames)
        {
            float distance = Vector4.Distance(objectColor, colorPair.Key);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestColorName = colorPair.Value;
            }
        }
        return closestColorName;
    }

    // Start is called before the first frame update
    void Start()
    {
        

        Object_Tag_List = FetchUniqueTagsInScene();
        // print all tags
        Debug.Log("Tags in Scene: " + string.Join(", ", Object_Tag_List));

        Certain_Object[] specified_Objects = new Certain_Object[MAX_OBJECTS];
        Scene_Objects scene_Objects = new Scene_Objects();
        int Total_Num = 0;

        for (int i = 0;i < Object_Tag_List.Count; i++)
        {
            GameObject[] Tag_Objects= GameObject.FindGameObjectsWithTag(Object_Tag_List[i]);
            

            for (int j = 0;  j < Tag_Objects.Length; j++)
            {
                Tag_Objects[j].name = Object_Tag_List[i] + "_" + (j + 1);
                Transform t = Tag_Objects[j].transform;
                

                //save to json

                var object_Information = new Certain_Object();
                //object_Information.Tag = Object_Tag_List[i];
                object_Information.Tag = Object_Tag_List[i].Replace("custom_", "");
                object_Information.Name = Object_Tag_List[i].Replace("custom_", "") + "_" +(j + 1);
                object_Information.Position = t.position;
                object_Information.Rotation = t.rotation;
                object_Information.LocalPosition = t.localPosition;
                object_Information.EulerAngles = t.eulerAngles;

                // Object material and color
                Renderer renderer = Tag_Objects[j].GetComponent<Renderer>();
                //if (renderer != null)
                if (renderer != null && renderer.material != null)
                {
                    object_Information.ObjectMaterial_string = renderer.material.name;
                    object_Information.ObjectColor_string = GetColorName(renderer.material.color);
                }
                else
                {
                    object_Information.ObjectMaterial_string = "Unknown Material";
                    object_Information.ObjectColor_string = "Unknown Color";
                }

                // Interactability - check if it has a collider and is not a trigger
                Collider collider = Tag_Objects[j].GetComponent<Collider>();
                object_Information.IsInteractable = collider != null && !collider.isTrigger;


                specified_Objects[Total_Num] = object_Information;
                

                Total_Num++;
            }
        }
        Debug.Log("Total object Num: " + Total_Num);


        Certain_Object[] object_List = new Certain_Object[Total_Num];
        for (int i = 0; i < Total_Num; i++)
        {
            object_List[i] = specified_Objects[i]; // The length of specified_Objects is 1000, and the length of object_List is the total number of objects
        }

        Scene_Objects all_info = new Scene_Objects();
        all_info.scene_Objects_List = object_List;
        string json_scene_objects = JsonUtility.ToJson(all_info);

        // Define the path to the scene_information folder and the file
        string directoryPath = ".\\scene_information";
        string filePath = directoryPath + "\\object_information_json.json";

        // Check if the directory exists
        if (!Directory.Exists(directoryPath))
        {
            // If the directory does not exist, create it
            Directory.CreateDirectory(directoryPath);
            Debug.Log("Directory created: " + directoryPath);
        }

        File.WriteAllText(filePath, json_scene_objects);
        Debug.Log("JSON data saved to: " + filePath);

        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
