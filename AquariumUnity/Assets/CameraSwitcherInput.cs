using UnityEngine;
using UnityEngine.InputSystem;

public class CameraSwitcherInput : MonoBehaviour
{
    public Camera cameraFront;
    public Camera cameraTop;

    private Controls controls;
    private bool vueFace = true;

    void Awake()
    {
        controls = new Controls();
        controls.ChangeView.SwitchCamera.performed += ctx => Switch();
    }

    void OnEnable()
    {
        controls.Enable();
    }

    void OnDisable()
    {
        controls.Disable();
    }

    void Switch()
    {
        vueFace = !vueFace;
        cameraFront.enabled = vueFace;
        cameraTop.enabled = !vueFace;
    }
}
