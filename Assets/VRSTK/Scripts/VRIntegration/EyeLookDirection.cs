using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ViveSR.anipal.Eye;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Detects which objects users are looking at (Eye Tracking) and sends out events with how long they looked at it and the direction with postion of this detected Object.</summary>
            public class EyeLookDirection : MonoBehaviour
            {
                public Telemetry.Event lookEvent;
                public Telemetry.Event eyeLookAtEvent;

                private GameObject lookingAt;

                //RaycastHits for Sphere and Object Collider
                private RaycastHit hitSphere;
                private float hitTimeSphere;

                private RaycastHit hitObjects;
                private float hitTimeObejcts;

                //GameObject for get the Sphere Collider radius
                private GameObject sphereEyeCollider;

                //Variable for enable Object Tracking
                public bool enableEyeObjectTracking = false;

                //Material for a LineRenderer
                public bool debugEyeTrackingPositionsWithLine;
                private Material lineRendererMaterial;
                private GameObject lineLeft;


                //The current eyedata
                private EyeData EyeData = new EyeData();
                public Vector3 eyeHitpoint;
                //The world eye direction
                public Vector3 eyeDirection;

                //True if the given eye tracking points are valid and up-to-date
                public bool validTracking = false;
                //Variable for last Error
                private ViveSR.Error lastError;

                void Start()
                {
                    Debug.Log("EyeLookDirection script started");

                    sphereEyeCollider = GameObject.Find("SphereEyeCollider");

                    lineLeft = new GameObject();
                    lineLeft.SetActive(false);
                    lineLeft.AddComponent<LineRenderer>();
                    lineLeft.GetComponent<LineRenderer>().enabled = false;
                    lineRendererMaterial = new Material(Shader.Find("Specular"));

                    //Initialize|Instancing off eye tracking 
                    if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING)
                    {
                        Debug.LogWarning("SRanipal not running (Status " + SRanipal_Eye_Framework.Status + "). Trying to (re)initialise");
                        var sranipal = SRanipal_Eye_Framework.Instance;
                        if (sranipal == null)
                        {
                            //The Framework script should be included in the scene
                            //If not, as a fallback, create a component
                            Debug.LogWarning("SRanipal_Eye_Framework should be included in world.");
                            sranipal = gameObject.AddComponent<SRanipal_Eye_Framework>();
                        }
                        sranipal.StartFramework();
                    }
                }

                // Update is called once per frame
                void Update()
                {
                    if (!debugEyeTrackingPositionsWithLine)
                    {
                        lineLeft.SetActive(false);
                        lineLeft.GetComponent<LineRenderer>().enabled = false;
                    }

                    var error = SRanipal_Eye_API.GetEyeData(ref EyeData);
                    var newData = SRanipal_Eye.GetVerboseData(out EyeData.verbose_data);

                    //No new data received from camera sensor - skip step
                    if (!newData)
                        return;

                    //Show error only once and not every frame
                    if (error != ViveSR.Error.WORK && lastError != error)
                    {
                        Debug.LogError("An error happened: " + error.ToString());
                    }
                    lastError = error;

                    var leftEye = EyeData.verbose_data.left.gaze_direction_normalized;
                    var rightEye = EyeData.verbose_data.right.gaze_direction_normalized;

                    //With this conditions only one eye is tracked (here is most the left Eye)!
                    if (leftEye != Vector3.zero)
                    {
                        this.validTracking = true;
                        CalculateWorldSpace(leftEye);
                    }
                    else if (rightEye != Vector3.zero)
                    {
                        this.validTracking = true;
                        CalculateWorldSpace(rightEye);
                    }
                    else
                    {
                        this.validTracking = false;
                    }

                }

                private void OnLookStart()
                {
                    if (lookingAt != null)
                    {
                        OnLookEnd();
                    }
                    lookingAt = hitObjects.transform.gameObject;
                    hitTimeObejcts = TestStage.GetTime();
                }

                private void OnLookEnd()
                {
                    float duration = TestStage.GetTime() - hitTimeObejcts;
                    GetComponent<EventSender>().SetEventValue("ObjectName", lookingAt.name);
                    GetComponent<EventSender>().SetEventValue("Duration", duration);
                    GetComponent<EventSender>().SetEventValue("EyeHitPoint", eyeHitpoint);
                    GetComponent<EventSender>().SetEventValue("EyeDirection", eyeDirection);
                    Debug.Log("lookingAt.name:=(" + lookingAt.name + ") \n eyeHitpoint:=(" + eyeHitpoint + ") \n eyeDirection=(" + eyeDirection + ") \n Duration=(" + duration + ")");

                    GetComponent<EventSender>().Deploy();
                    lookingAt = null;
                }

                public GameObject getLookingAt()
                {
                    return lookingAt;
                }

                private void SphereColliderEventSender()
                {
                    float duration = TestStage.GetTime() - hitTimeSphere;

                    // Disable the spring on all HingeJoints in this game object
                    List<EventSender> eventSender = new List<EventSender>();

                    GetComponents(eventSender);
                    // EyeLookAtObeject
                    eventSender[0].SetEventValue("ObjectName", gameObject.name);
                    eventSender[0].SetEventValue("EyeHitPoint", eyeHitpoint);
                    eventSender[0].SetEventValue("EyeDirection", eyeDirection);
                    eventSender[0].Deploy();

                    // EyeLookAt
                    eventSender[1].SetEventValue("position_Transform", transform.position);
                    eventSender[1].SetEventValue("rotation_Transform", Quaternion.LookRotation(eyeDirection));
                    eventSender[1].SetEventValue("Duration", duration);
                    eventSender[1].Deploy();

                    Debug.Log("name:=(" + gameObject.name + ") \n eyeHitpoint:=(" + eyeHitpoint + ") \n eyeDirection=(" + eyeDirection + ") \n Duration=(" + duration + ")");
                }

                /**
                * Calculates the world direction and updates the "LookAt" position
                */
                void CalculateWorldSpace(Vector3 direction)
                {
                    //The direction from sr_anipal is in "raw" world space
                    //So no translation or rotation of the head is taken in account
                    //This will translate it to the location and rotation of the world space
                    //direction = head.transform.TransformDirection(direction);
                    direction = transform.TransformDirection(direction);

                    //The data we get from sr_anipal is wrongly coded
                    //When looking up right, we get the direction up left
                    //Instead of down left, we get down right
                    //So we switch to 2d space (using x and z but not the height y)
                    //Then we get the angle between the cameras lookat and the eye lookat
                    //We negate the angle, multipliying by two (lookat -> center -> lookat) and
                    // use the quaternion transform to get the "real" data.
                    var eyeDirection2 = new Vector2(direction.x, direction.z);
                    //var headDirection2 = new Vector2(head.transform.forward.x, head.transform.forward.z);
                    var headDirection2 = new Vector2(transform.forward.x, transform.forward.z);
                    var correctedDirection3 = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * direction;
                    direction = correctedDirection3;

                    this.eyeDirection = direction;

                    RayToSphereColliderCast(direction);

                    if (enableEyeObjectTracking)
                        RayToObjectColliderCast(direction);
                }

                void RayToSphereColliderCast(Vector3 direction)
                {
                    int layerMask = 1 << 8;
                    layerMask = ~layerMask;

                    sphereEyeCollider.GetComponent<MeshCollider>().enabled = true;
                    //Get radius of Sphere Collider
                    float radius = sphereEyeCollider.transform.lossyScale.x / 2.0f;
                    Debug.Log("sphere.radius:" + radius);

                    hitTimeSphere = TestStage.GetTime();

                    if (Physics.Raycast(transform.position, direction, out hitSphere, radius, layerMask))
                    {
                        Debug.Log("Hit the Sphere Collider with direction:" + direction + " hitPoint:" + hitSphere.point);
                        //When hit collider: Use collider as hitpoint
                        this.eyeHitpoint = hitSphere.point;
                        DrawLine(transform.position, eyeHitpoint, Color.green);
                    }
                    else
                    {
                        Debug.Log("No hit on the Sphere Collider!");
                        this.eyeHitpoint = direction.normalized * 100;

                        DrawLine(transform.position, direction, Color.red);
                    }

                    if (hitSphere.transform != null)
                        SphereColliderEventSender();
                }

                void RayToObjectColliderCast(Vector3 direction)
                {
                    int layerMask = 1 << 8;
                    layerMask = ~layerMask;

                    sphereEyeCollider.GetComponent<MeshCollider>().enabled = false;

                    if (Physics.Raycast(transform.position, direction, out hitObjects, Mathf.Infinity, layerMask))
                    {
                        Debug.Log("Hit the Collider of an Object, with direction:" + direction + " and hitPoint:" + hitObjects.point);
                        //When hit collider: Use collider as hitpoint
                        this.eyeHitpoint = hitObjects.point;
                        DrawLine(transform.position, eyeHitpoint, Color.red);
                    }
                    else
                    {
                        Debug.Log("No hit on an Object Collider!");
                        this.eyeHitpoint = direction.normalized * 100;
                    }

                    //With out Sphere-Collider
                    if (hitObjects.transform != null && lookingAt != hitObjects.transform.gameObject)
                    {
                        OnLookStart();
                    }
                    else if (hitObjects.transform == null && lookingAt != null)
                    {
                        OnLookEnd();
                    }
                }

                //This method is for testing the plausibility for eyetracking positions
                //Line will be drawed by start and end pos. with given color 
                void DrawLine(Vector3 start, Vector3 end, Color color)
                {
                    if (debugEyeTrackingPositionsWithLine)
                    {
                        lineLeft.transform.position = start;
                        LineRenderer lr = lineLeft.GetComponent<LineRenderer>();
                        lineRendererMaterial.SetColor("_Color", color);
                        lr.material = lineRendererMaterial;
                        lr.startColor = color;
                        lr.endColor = color;
                        lr.startWidth = 0.01f;
                        lr.endWidth = 0.01f;
                        lr.SetPosition(0, start);
                        lr.SetPosition(1, end);

                        lineLeft.SetActive(true);
                        lineLeft.GetComponent<LineRenderer>().enabled = true;
                    }
                }
            }
        }
    }
}