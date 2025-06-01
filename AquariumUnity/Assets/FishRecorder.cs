using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

[RequireComponent(typeof(Poisson))]
public class FishRecorder : MonoBehaviour
{
    [Header("Recording Settings")]
    public float recordingInterval = 0.1f;
    public bool autoStartRecording = true;
    public string defaultSaveName = "fish_recording";
    public bool autoSaveOnExit = true;
    public bool loadRecording = false;
    public string fileName = "";

    // Serialized state structure
    [System.Serializable]
    public struct FishState
    {
        public float[] position; // [x,y,z]
        public float[] rotation; // [x,y,z,w]
        public float timestamp;
    }

    [System.Serializable]
    private class FishRecording
    {
        public List<FishState> states = new List<FishState>();
        public float interval;
        public DateTime creationDate;
        public string fishName;
    }

    private Vector3 originalScale;

    [Header("Dimensions souhaitées du poisson (en unités)")]
    public float longueurZ = 1.0f;
    public float hauteurY = 0.4f;
    public float largeurX = 0.3f;

    private Poisson fish;
    private List<FishState> recordedStates = new List<FishState>();
    private float timeSinceLastRecord;
    private bool isRecording;
    private bool isReplaying;
    private int replayIndex;
    private float replayStartTime;

    [Header("Auto-Save Settings")]
    [Tooltip("Only auto-save if we have at least this many recorded states")]
    public int minStatesForAutoSave = 10;

    void Start()
    {
        fish = GetComponent<Poisson>();

        if(loadRecording)
        {
            autoStartRecording = false;
            autoSaveOnExit = false;

        }

        if (autoStartRecording)
        {
            StartRecording();
        }
        if (loadRecording)
        {
            AjusterTaillePoisson();
            originalScale = transform.localScale; // Store initial scale

            LoadRecording(fileName);
            StartReplay();
        }
    }

    void Update()
    {
        if (isRecording)
        {
            timeSinceLastRecord += Time.deltaTime;
            if (timeSinceLastRecord >= recordingInterval)
            {
                RecordState();
                timeSinceLastRecord = 0f;
            }
        }
        else if (isReplaying)
        {
            UpdateReplay();
        }
    }

    void AjusterTaillePoisson()
    {
        Renderer rend = GetComponentInChildren<Renderer>();
        if (!rend) return;

        Vector3 tailleMesh = rend.bounds.size;
        if (tailleMesh == Vector3.zero) return;

        Vector3 facteur = new Vector3(
            largeurX / tailleMesh.x,
            hauteurY / tailleMesh.y,
            longueurZ / tailleMesh.z
        );

        transform.localScale = facteur;
    }

    // ==================== Recording Controls ====================
    public void StartRecording()
    {
        recordedStates.Clear();
        isRecording = true;
        isReplaying = false;
        RecordState();
        Debug.Log("Started recording fish movement");
    }

    public void StopRecording()
    {
        isRecording = false;
        Debug.Log($"Stopped recording. Captured {recordedStates.Count} states");
    }

    // ==================== Replay Controls ====================
    public void StartReplay()
    {
        if (recordedStates.Count == 0)
        {
            Debug.LogWarning("No recording to replay");
            return;
        }

        replayIndex = 0;
        isReplaying = true;
        isRecording = false;
        replayStartTime = Time.time;
        ApplyState(recordedStates[0]);

        fish.enabled = false;
        Debug.Log("Started replay");
    }

    public void StopReplay()
    {
        isReplaying = false;
        fish.enabled = true;
        Debug.Log("Stopped replay");
    }

    // ==================== State Management ====================
    private void RecordState()
    {
        recordedStates.Add(new FishState
        {
            position = new float[] {
                fish.transform.position.x,
                fish.transform.position.y,
                fish.transform.position.z
            },
            rotation = new float[] {
                fish.transform.rotation.x,
                fish.transform.rotation.y,
                fish.transform.rotation.z,
                fish.transform.rotation.w
            },
            timestamp = Time.time
        });
    }

    private void UpdateReplay()
    {
        float currentReplayTime = Time.time - replayStartTime;

        while (replayIndex < recordedStates.Count - 2 &&
               recordedStates[replayIndex + 1].timestamp <= currentReplayTime)
        {
            replayIndex++;
        }

        if (replayIndex >= recordedStates.Count - 1)
        {
            StopReplay();
            return;
        }

        FishState current = recordedStates[replayIndex];
        FishState next = recordedStates[replayIndex + 1];

        float segmentDuration = next.timestamp - current.timestamp;
        float segmentProgress = currentReplayTime - current.timestamp;
        float t = Mathf.Clamp01(segmentProgress / segmentDuration);

        fish.transform.position = Vector3.Lerp(
            new Vector3(current.position[0], current.position[1], current.position[2]),
            new Vector3(next.position[0], next.position[1], next.position[2]),
            t
        );

        fish.transform.rotation = Quaternion.Slerp(
            new Quaternion(current.rotation[0], current.rotation[1], current.rotation[2], current.rotation[3]),
            new Quaternion(next.rotation[0], next.rotation[1], next.rotation[2], next.rotation[3]),
            t
        );
    }

    private void ApplyState(FishState state)
    {
        fish.transform.position = new Vector3(state.position[0], state.position[1], state.position[2]);
        fish.transform.rotation = new Quaternion(state.rotation[0], state.rotation[1], state.rotation[2], state.rotation[3]);

        // Maintain original scale
        fish.transform.localScale = originalScale;
    }

    void OnApplicationQuit()
    {
        AutoSaveIfEnabled();
    }

    // Called when exiting play mode in the editor
    void OnDestroy()
    {
        if (Application.isPlaying)
        {
            AutoSaveIfEnabled();
        }
    }

    private void AutoSaveIfEnabled()
    {
        if (autoSaveOnExit && isRecording && recordedStates.Count >= minStatesForAutoSave)
        {
            string autoSaveName = $"{defaultSaveName}_autosave_{DateTime.Now:yyyyMMdd_HHmmss}";
            SaveRecording(autoSaveName);
            Debug.Log($"Auto-saved recording as: {autoSaveName}");
        }
    }

    // ==================== Save/Load Functionality ====================
    public void SaveRecording(string fileName = null)
    {
        if (recordedStates.Count == 0)
        {
            Debug.LogWarning("Nothing to save - no recorded states");
            return;
        }

        string saveName = string.IsNullOrEmpty(fileName) ? defaultSaveName : fileName;
        string path = Path.Combine(Application.persistentDataPath, saveName + ".json");

        FishRecording recording = new FishRecording
        {
            states = recordedStates,
            interval = recordingInterval,
            creationDate = DateTime.Now,
            fishName = fish.gameObject.name
        };

        string json = JsonUtility.ToJson(recording, true);
        File.WriteAllText(path, json);

        Debug.Log($"Saved recording to: {path}\n{recordedStates.Count} states");
    }

    public bool LoadRecording(string fileName = null)
    {
        string saveName = string.IsNullOrEmpty(fileName) ? defaultSaveName : fileName;
        string path = Path.Combine(Application.persistentDataPath, saveName + ".json");

        if (!File.Exists(path))
        {
            Debug.LogWarning($"No recording found at: {path}");
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            FishRecording recording = JsonUtility.FromJson<FishRecording>(json);

            recordedStates = recording.states;
            recordingInterval = recording.interval;

            Debug.Log($"Loaded recording: {recording.fishName}\n" +
                     $"Created: {recording.creationDate}\n" +
                     $"{recordedStates.Count} states");

            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load recording: {e.Message}");
            return false;
        }
    }

    // ==================== Utility Methods ====================
    public int GetRecordedStateCount() => recordedStates.Count;
    public bool IsRecording() => isRecording;
    public bool IsReplaying() => isReplaying;

    public string[] GetSavedRecordings()
    {
        if (!Directory.Exists(Application.persistentDataPath))
        {
            return new string[0];
        }

        List<string> recordings = new List<string>();
        foreach (string file in Directory.GetFiles(Application.persistentDataPath, "*.json"))
        {
            try
            {
                string json = File.ReadAllText(file);
                FishRecording recording = JsonUtility.FromJson<FishRecording>(json);
                recordings.Add(Path.GetFileNameWithoutExtension(file) + $" ({recording.fishName}, {recording.states.Count} states)");
            }
            catch
            {
                // Skip corrupted files
            }
        }
        return recordings.ToArray();
    }
}