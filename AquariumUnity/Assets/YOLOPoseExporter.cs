using UnityEngine;
using System.IO;
using System.Text;
using System.Collections.Generic;

public class YOLOPoseExporter : MonoBehaviour
{
    [Header("Settings")]
    public Camera exportCamera;
    public bool capturePose = true;
    public bool captureFrame = false;
    public string outputFolder = "D:\\UnityProjects\\AquariumUnity\\dataset\\";

    [Header("Keypoints")]
    public List<Transform> sardines;
    public List<Transform> keypoints;

    private int frameCount = 1;
    private RenderTexture renderTexture;
    private Texture2D screenshot;
    private int imageWidth = 3840;
    private int imageHeight = 2160;

    void Start()
    {
        string labelFolderPath = outputFolder + "\\labels";

        if (!Directory.Exists(labelFolderPath))
        {
            Directory.CreateDirectory(labelFolderPath);
        }

        string imagesFolderPath =outputFolder + "\\images";
        if (!Directory.Exists(imagesFolderPath))
        {
            Directory.CreateDirectory(imagesFolderPath);
        }

        // Setup render texture for screenshots
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        screenshot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        if (exportCamera == null) exportCamera = Camera.main;
        Directory.CreateDirectory(outputFolder);
    }

    void Update()
    {
        if (frameCount == 5000) UnityEditor.EditorApplication.isPlaying = false;
        if (capturePose) ExportYoloPoseData();
        if (captureFrame) CaptureFrame();
        frameCount++;
    }

    public void ExportYoloPoseData()
    {
        foreach (Transform sardine in sardines)
        {
            StringBuilder yoloLine = new StringBuilder();
            yoloLine.Append("0 "); // Class 0

            Rect boudingBox = getBoundingBox(sardine);
            yoloLine.Append($"{boudingBox.x} {boudingBox.y} {boudingBox.width} {boudingBox.height} ");

            foreach (Transform keypoint in sardine)
            {
                if (keypoint.name == "Armature" || keypoint.name == "SardineSkin") continue;
                AppendKeypoint(yoloLine, keypoint);
            }

            // Save to file
            string filePath = Path.Combine(outputFolder + "\\labels", $"F_{(frameCount):0000}.txt");
            string noCommaYoloLine = yoloLine.ToString().Replace(",", ".").Trim();
            if (File.Exists(filePath))
            {
                File.AppendAllText(filePath, "\n" + noCommaYoloLine);
            }
            else
            {
                File.WriteAllText(filePath, noCommaYoloLine);
            }
        }
    }

    //private void AppendKeypoint(StringBuilder sb, Transform keypoint)
    //{
    //    if (keypoint == null)
    //    {
    //        sb.Append("0 0 1 ");
    //        return;
    //    }

    //    Vector3 viewportPoint = exportCamera.WorldToViewportPoint(keypoint.position);

    //    // Flip Y-coordinate for YOLO format
    //    float yoloY = 1.0f - viewportPoint.y;
    //    sb.Append($"{viewportPoint.x:F6} {yoloY:F6} ");
    //}

    //Rect getBoundingBox(Transform transform)
    //{
    //    Renderer renderer = transform.GetComponentInChildren<Renderer>();
    //    if (renderer == null)
    //        return new Rect(0, 0, 0, 0);

    //    Bounds bounds = renderer.bounds;

    //    // Get all 8 corners of the bounding box
    //    Vector3[] corners = new Vector3[8];
    //    corners[0] = new Vector3(bounds.min.x, bounds.min.y, bounds.min.z);
    //    corners[1] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
    //    corners[2] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
    //    corners[3] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
    //    corners[4] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
    //    corners[5] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
    //    corners[6] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);
    //    corners[7] = new Vector3(bounds.max.x, bounds.max.y, bounds.max.z);

    //    // Convert all corners to viewport coordinates
    //    Vector3[] viewportCorners = new Vector3[8];
    //    for (int i = 0; i < 8; i++)
    //    {
    //        viewportCorners[i] = exportCamera.WorldToViewportPoint(corners[i]);
    //    }

    //    // Find the actual min/max in viewport space
    //    float minX = float.MaxValue, maxX = float.MinValue;
    //    float minY = float.MaxValue, maxY = float.MinValue;

    //    foreach (Vector3 corner in viewportCorners)
    //    {
    //        if (corner.x < minX) minX = corner.x;
    //        if (corner.x > maxX) maxX = corner.x;
    //        if (corner.y < minY) minY = corner.y;
    //        if (corner.y > maxY) maxY = corner.y;
    //    }

    //    // Calculate center and size in YOLO format
    //    float centerX = (minX + maxX) * 0.5f;
    //    float centerY = (minY + maxY) * 0.5f;
    //    float width = maxX - minX;
    //    float height = maxY - minY;

    //    // Flip Y coordinate for YOLO format (YOLO has origin at top-left)
    //    centerY = 1.0f - centerY;

    //    return new Rect(centerX, centerY, width, height);
    //}

    private void AppendKeypoint(StringBuilder sb, Transform keypoint)
    {
        if (keypoint == null)
        {
            sb.Append("0 0 1 ");
            return;
        }
        Vector3 viewportPoint = exportCamera.WorldToViewportPoint(keypoint.position);

        // Clamp values to 0-1 range for YOLO format
        float clampedX = Mathf.Clamp01(viewportPoint.x);
        float clampedY = Mathf.Clamp01(viewportPoint.y);

        // Flip Y-coordinate for YOLO format
        float yoloY = 1.0f - clampedY;

        sb.Append($"{clampedX:F6} {yoloY:F6} ");
    }

    Rect getBoundingBox(Transform transform)
    {
        Renderer renderer = transform.GetComponentInChildren<Renderer>();
        if (renderer == null)
            return new Rect(0, 0, 0, 0);

        Bounds bounds = renderer.bounds;

        // Get all 8 corners of the bounding box
        Vector3[] corners = new Vector3[8];
        corners[0] = new Vector3(bounds.min.x, bounds.min.y, bounds.min.z);
        corners[1] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
        corners[2] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
        corners[3] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
        corners[4] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
        corners[5] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
        corners[6] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);
        corners[7] = new Vector3(bounds.max.x, bounds.max.y, bounds.max.z);

        // Convert all corners to viewport coordinates and clamp them
        Vector3[] viewportCorners = new Vector3[8];
        for (int i = 0; i < 8; i++)
        {
            Vector3 viewportPoint = exportCamera.WorldToViewportPoint(corners[i]);
            viewportCorners[i] = new Vector3(
                Mathf.Clamp01(viewportPoint.x),
                Mathf.Clamp01(viewportPoint.y),
                viewportPoint.z
            );
        }

        // Find the actual min/max in viewport space
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;

        foreach (Vector3 corner in viewportCorners)
        {
            if (corner.x < minX) minX = corner.x;
            if (corner.x > maxX) maxX = corner.x;
            if (corner.y < minY) minY = corner.y;
            if (corner.y > maxY) maxY = corner.y;
        }

        // Calculate center and size in YOLO format
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        float width = maxX - minX;
        float height = maxY - minY;

        // Flip Y coordinate for YOLO format (YOLO has origin at top-left)
        centerY = 1.0f - centerY;

        // Ensure final values are within valid range
        centerX = Mathf.Clamp01(centerX);
        centerY = Mathf.Clamp01(centerY);
        width = Mathf.Clamp01(width);
        height = Mathf.Clamp01(height);

        return new Rect(centerX, centerY, width, height);
    }

    void CaptureFrame()
    {
        string frameName = $"F_{frameCount:0000}";

        // Capture image
        CaptureImage(frameName);
    }

    void CaptureImage(string frameName)
    {
        // Set camera target texture
        RenderTexture previousRT = exportCamera.targetTexture;
        exportCamera.targetTexture = renderTexture;

        // Render camera
        exportCamera.Render();

        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        screenshot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenshot.Apply();

        // Save image
        byte[] imageData = screenshot.EncodeToJPG();
        string imagePath = Path.Combine(outputFolder + "\\images", frameName + ".jpg");
        File.WriteAllBytes(imagePath, imageData);

        // Restore camera settings
        exportCamera.targetTexture = previousRT;
        RenderTexture.active = null;
    }

}