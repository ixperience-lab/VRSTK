﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ViveSR.anipal.Eye;

namespace STK
{
    ///<summary>Detects which objects users are looking at (Eye Tracking) and sends out events with how long they looked at it and the direction with postion of this detected Object.</summary>
    public class STKEyeLookDirection : MonoBehaviour
    {
        public STKEvent lookEvent;
        private GameObject lookingAt;

        private RaycastHit hit;
        private float hitTime;

        //Material for a LineRenderer
        public bool debugEyeTrackingPositionsWithLine;
        private Material lineRendererMaterial;
        private GameObject lineLeft;
        //private GameObject lineRight;

        //The current eyedata
        private EyeData EyeData = new EyeData();
        public Vector3 eyeHitpoint;
        //The world eye direction
        public Vector3 eyeDirection;

        // --- test wh
        //The vr camera
        //public GameObject head;
        //The world eye direction
        //public Vector3 eyeLeftDirection;
        //The world hitpoint is available, else the point 100 meter towards
        //public Vector3 eyeLeftHitpoint;
        //The world eye direction
        //public Vector3 eyeRightDirection;
        //The world hitpoint is available, else the point 100 meter towards
        //public Vector3 eyeRightHitpoint;
        // ---

        //True if the given eye tracking points are valid and up-to-date
        public bool validTracking = false;
        //Variable for last Error
        private ViveSR.Error lastError;


        void Start()
        {
            Debug.LogWarning("STKEyeLookDirection script started");
            lineLeft = new GameObject();
            lineLeft.AddComponent<LineRenderer>();
            //lineRight = new GameObject();
            //lineRight.AddComponent<LineRenderer>();
            lineLeft.GetComponent<LineRenderer>().enabled = false; 
            //lineRight.GetComponent<LineRenderer>().enabled = false;
            lineRendererMaterial = new Material(Shader.Find("Specular"));

            //Initialize|Instancing off eye tracking 
            if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING)
            {
                Debug.LogError("SRanipal not running (Status " + SRanipal_Eye_Framework.Status + "). Trying to (re)initialise");
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
            if (debugEyeTrackingPositionsWithLine)
            {
                lineLeft.SetActive(true);
                //lineRight.SetActive(true);
                lineLeft.GetComponent<LineRenderer>().enabled = true;
                //lineRight.GetComponent<LineRenderer>().enabled = true;
            }
            else
            {
                lineLeft.SetActive(false);
                //lineRight.SetActive(false);
                lineLeft.GetComponent<LineRenderer>().enabled = false;
                //lineRight.GetComponent<LineRenderer>().enabled = false;
            }

                var error = SRanipal_Eye_API.GetEyeData(ref EyeData);
            var newData = SRanipal_Eye.GetVerboseData(out EyeData.verbose_data);

            //Debug.LogError("SRanipal (Status " + SRanipal_Eye_Framework.Status + ")");
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

            // if (leftEye != Vector3.zero && rightEye != Vector3.zero)
            // Debug.Log("Eyes: LEFT:=" + leftEye + ", RIGHT:=" + rightEye);

            // with this conditions only one eye tracked (here the most the left Eye)!
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
            lookingAt = hit.transform.gameObject;
            hitTime = STKTestStage.GetTime();
        }

        private void OnLookEnd()
        {
            float duration = STKTestStage.GetTime() - hitTime;
            GetComponent<STKEventSender>().SetEventValue("ObjectName", lookingAt.name);
            GetComponent<STKEventSender>().SetEventValue("Duration", duration);
            GetComponent<STKEventSender>().SetEventValue("EyeHitPoint", eyeHitpoint);
            GetComponent<STKEventSender>().SetEventValue("EyeDirection", eyeDirection);
            Debug.Log("lookingAt.name:=(" + lookingAt.name + ") \n eyeHitpoint:=(" + eyeHitpoint + ") \n eyeDirection=(" + eyeDirection + ") \n Duration=(" + duration + ")");
            //GetComponent<STKEventSender>().SetEventValue("EyeRightHitPoint", eyeRightHitpoint);
            //GetComponent<STKEventSender>().SetEventValue("EyeRightDirection", eyeRightDirection);
            //Debug.Log("lookingAt.name:=(" + lookingAt.name + ") \n eyeLeftHitpoint:=(" + eyeLeftHitpoint + ") \n eyeLeftDirection=(" + eyeLeftDirection + ") \n eyeRightHitpoint:=(" + eyeRightHitpoint + ") \n eyeRightDirection=(" + eyeRightDirection + ") \n Duration=(" + duration + ")");

            GetComponent<STKEventSender>().Deploy();
            lookingAt = null;
        }

        public GameObject getLookingAt()
        {
            return lookingAt;
        }

        /**
        * Calculates the world direction and updates the "LookAt" position
        */
        void CalculateWorldSpace(Vector3 direction)
        {
            // Bit shift the index of the layer (8) to get a bit mask
            int layerMask = 1 << 8;
            layerMask = ~layerMask;

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
            var correctedDirecion3 = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * direction;
            direction = correctedDirecion3;

            this.eyeDirection = direction;

            //Debug.Log("looked in direction:=" + direction);

            //Physics.SphereCast(transform.position, 0.2f, transform.forward, out hit, 100);
            //RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, Mathf.Infinity, layerMask))//if (Physics.Raycast(head.transform.position, direction, out hit, Mathf.Infinity, layerMask))
            {
                //When hit collider: Use collider as hitpoint
                this.eyeHitpoint = hit.point;
                //Debug.DrawLine(transform.position, eyeHitpoint, Color.green, 0.2f);
                if (debugEyeTrackingPositionsWithLine)
                    DrawLine(transform.position, eyeHitpoint, Color.green, lineLeft);
            }
            else
            {
                //When not hit: Draw line 100 meters
                this.eyeHitpoint = direction.normalized * 100;
            }

            
            // WH implementation
            //if (isLeftEye)
            //{
            //    this.eyeLeftHitpoint = this.eyeHitpoint;
            //    this.eyeLeftDirection = direction;
            //    //Debug.DrawLine(transform.position, eyeHitpoint, Color.green, 0.2f);
            //    if (debugEyeTrackingPositionsWithLine)
            //        DrawLine(transform.position, eyeHitpoint, Color.green, lineLeft);
            //}
            //else
            //{
            //    this.eyeRightHitpoint = this.eyeHitpoint;
            //    this.eyeRightDirection = direction;
            //    //Debug.DrawLine(transform.position, eyeHitpoint, Color.red, 0.2f);
            //    if (debugEyeTrackingPositionsWithLine)
            //        DrawLine(transform.position, eyeHitpoint, Color.red, lineRight);
            //}
            
            if (hit.transform != null && lookingAt != hit.transform.gameObject)
            {
                OnLookStart();
            }
            else if (hit.transform == null && lookingAt != null)
            {
                OnLookEnd();
            }

        }


        //This method is for testing the plausibility for eyetracking positions
        //Line will be drawed by start and end pos. with given color 
        void DrawLine(Vector3 start, Vector3 end, Color color, GameObject line)
        {
            line.transform.position = start;
            LineRenderer lr = line.GetComponent<LineRenderer>();
            lineRendererMaterial.SetColor("_Color", color);
            lr.material = lineRendererMaterial;
            lr.SetColors(color, color);
            lr.SetWidth(0.01f, 0.01f);
            lr.SetPosition(0, start);
            lr.SetPosition(1, end);
        }
    }
}