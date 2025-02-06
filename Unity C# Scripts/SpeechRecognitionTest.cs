using System.IO;
using HuggingFace.API;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using System.Collections;

public class SpeechRecognitionTest : MonoBehaviour
{
    [SerializeField] private Button startButton;
    [SerializeField] private Button stopButton;
    [SerializeField] private TextMeshProUGUI InputText;  // display speech recognition result
    [SerializeField] private TextMeshProUGUI OutputText; // display server reply

    private AudioClip clip;
    private byte[] bytes;
    private bool recording;

    private string serverUrl = "";  // server
    private string timeServerUrl = "";    // send response_time to server


    private void Start()
    {
        startButton.onClick.AddListener(StartRecording);
        stopButton.onClick.AddListener(StopRecording);
        stopButton.interactable = false;
    }

    private void Update()
    {
        if (recording && Microphone.GetPosition(null) >= clip.samples)
        {
            StopRecording();
        }
    }

    private void StartRecording()
    {
        InputText.text = "";  // clear input textbox
        OutputText.text = ""; // clear output textbox
        startButton.interactable = false;
        stopButton.interactable = true;
        clip = Microphone.Start(null, false, 10, 44100);
        recording = true;
    }

    private void StopRecording()
    {
        var position = Microphone.GetPosition(null);
        Microphone.End(null);
        var samples = new float[position * clip.channels];
        clip.GetData(samples, 0);
        bytes = EncodeAsWAV(samples, clip.frequency, clip.channels);
        recording = false;
        SendRecording();
    }

    private void SendRecording()
    {
        stopButton.interactable = false;
        HuggingFaceAPI.AutomaticSpeechRecognition(bytes, response =>
        {
            InputText.text = response;  // Displays speech-to-text results in InputText
            startButton.interactable = true;

            // send identifying text and camera data to the server
            StartCoroutine(SendPoseData(response));
        }, error =>
        {
            InputText.text = "Error: " + error; // display error information in InputText
            startButton.interactable = true;
        });
    }

    private byte[] EncodeAsWAV(float[] samples, int frequency, int channels)
    {
        using (var memoryStream = new MemoryStream(44 + samples.Length * 2))
        {
            using (var writer = new BinaryWriter(memoryStream))
            {
                writer.Write("RIFF".ToCharArray());
                writer.Write(36 + samples.Length * 2);
                writer.Write("WAVE".ToCharArray());
                writer.Write("fmt ".ToCharArray());
                writer.Write(16);
                writer.Write((ushort)1);
                writer.Write((ushort)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2);
                writer.Write((ushort)(channels * 2));
                writer.Write((ushort)16);
                writer.Write("data".ToCharArray());
                writer.Write(samples.Length * 2);

                foreach (var sample in samples)
                {
                    writer.Write((short)(sample * short.MaxValue));
                }
            }
            return memoryStream.ToArray();
        }
    }

    IEnumerator SendPoseData(string message)
    {
        // Get the Position and Rotation of the main camera
        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            Vector3 cameraPosition = mainCamera.transform.position;
            Quaternion cameraRotation = mainCamera.transform.rotation;

            // recorder start time
            float startTime = Time.time;

            // create JSON data
            CameraInfo cameraInfo = new CameraInfo
            {
                text = message,
                position = new Position { x = cameraPosition.x, y = cameraPosition.y, z = cameraPosition.z },
                rotation = new Rotation { x = cameraRotation.x, y = cameraRotation.y, z = cameraRotation.z, w = cameraRotation.w }
            };

            // use JsonUtility serielize object
            string json = JsonUtility.ToJson(cameraInfo);
            Debug.Log("json: " + json);

            UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
            byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(jsonToSend);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                // Record the end time and calculate the time spent
                float endTime = Time.time;
                float responseTime = endTime - startTime;
                Debug.Log("Data sent successfully: " + request.downloadHandler.text);
                OutputText.text = request.downloadHandler.text;  // display server replay in OutputText

                // Send time-consuming data separately
                StartCoroutine(SendResponseTime(responseTime));
            }
            else
            {
                Debug.LogError("Error sending data: " + request.error);
                OutputText.text = "Error sending to server!"; // display error information in OutputText
            }
        }
        else
        {
            Debug.LogError("Main camera not found!");
        }
    }

    IEnumerator SendResponseTime(float responseTime)
    {
        // create JSON data, only include response time
        ResponseTimeInfo timeInfo = new ResponseTimeInfo { responseTime = responseTime };
        string json = JsonUtility.ToJson(timeInfo);
        Debug.Log("Sending response time: " + json);

        UnityWebRequest request = new UnityWebRequest(timeServerUrl, "POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(jsonToSend);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Response time sent successfully: " + request.downloadHandler.text);
        }
        else
        {
            Debug.LogError("Error sending response time: " + request.error);
        }
    }

    // define class to store camera information
    [System.Serializable]
    public class CameraInfo
    {
        public string text;  // speech-to-text result
        public Position position;
        public Rotation rotation;
    }

    [System.Serializable]
    public class Position
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class Rotation
    {
        public float x;
        public float y;
        public float z;
        public float w;
    }

    // define class to store response time
    [System.Serializable]
    public class ResponseTimeInfo
    {
        public float responseTime; // Total time taken to request
    }
}


