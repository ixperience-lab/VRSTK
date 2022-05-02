    using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Route : MonoBehaviour
{
    [SerializeField]
    private Transform[] controlPoints;

    private Vector3 lastGizmoPositon;

    private Vector3 gizmoPosition;

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.grey;

        lastGizmoPositon = new Vector3(controlPoints[0].position.x, controlPoints[0].position.y, controlPoints[0].position.z);
        for (float t = 0; t <= 1; t += 0.05f)
        {
            
            gizmoPosition = Mathf.Pow(1 - t, 3) * controlPoints[0].position + 3 * Mathf.Pow(1 - t, 2) * t * controlPoints[1].position + 3 * (1 - t) * Mathf.Pow(t, 2) * controlPoints[2].position + Mathf.Pow(t, 3) * controlPoints[3].position;
            
            Gizmos.DrawLine(gizmoPosition, lastGizmoPositon);

            lastGizmoPositon = new Vector3(gizmoPosition.x, gizmoPosition.y, gizmoPosition.z);
        }

        Gizmos.DrawLine(new Vector3(controlPoints[0].position.x, controlPoints[0].position.y, controlPoints[0].position.z), new Vector3(controlPoints[1].position.x, controlPoints[1].position.y, controlPoints[1].position.z));

        Gizmos.DrawLine(new Vector3(controlPoints[2].position.x, controlPoints[2].position.y, controlPoints[2].position.z), new Vector3(controlPoints[3].position.x, controlPoints[3].position.y, controlPoints[3].position.z));
    }
}
