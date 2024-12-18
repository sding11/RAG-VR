using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 5f;       // player move speed
    public float rotationSpeed = 100f; // player rotation Speed
    public Transform cameraTransform; // camera Transform

    void Update()
    {
        MovePlayer();
        RotatePlayer();
    }

    void MovePlayer()
    {
        // Get the up and down arrow (W/S or ↑/↓) Input to move forward and backward
        float moveZ = Input.GetAxis("Vertical");

        // Get the camera orientation
        Vector3 forward = cameraTransform.forward;
        Vector3 right = cameraTransform.right;

        // Keep moving horizontally (ignoring the Y component)
        forward.y = 0;
        right.y = 0;
        forward.Normalize();
        right.Normalize();

        // Calculated direction of movement
        Vector3 moveDirection = forward * moveZ;

        // move player
        transform.Translate(moveDirection * moveSpeed * Time.deltaTime, Space.World);
    }

    void RotatePlayer()
    {
        // Get left and right arrows (A/D or ←/→) Input for horizontal rotation
        float rotationInput = Input.GetAxis("Horizontal");

        // Calculated rotation Angle
        float rotation = rotationInput * rotationSpeed * Time.deltaTime;

        // Rotate the player around the Y axis
        transform.Rotate(Vector3.up * rotation);
    }
}
