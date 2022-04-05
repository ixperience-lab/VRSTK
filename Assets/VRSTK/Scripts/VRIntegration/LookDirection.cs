using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Detects which objects users are looking at and sends out events with how long they looked at it.</summary>
            public class LookDirection : MonoBehaviour
            {

                public Telemetry.Event lookEvent;
                private GameObject lookingAt;

                private RaycastHit hit;
                private float hitTime;

                void Start()
                {

                }

                void Update()
                {
                    Physics.SphereCast(transform.position, 0.2f, transform.forward, out hit, 100);

                    if (hit.transform != null && lookingAt != hit.transform.gameObject)
                    {
                        OnLookStart();
                    }
                    else if (hit.transform == null && lookingAt != null)
                    {
                        OnLookEnd();
                    }
                }

                private void OnLookStart()
                {
                    if (lookingAt != null)
                    {
                        OnLookEnd();
                    }
                    lookingAt = hit.transform.gameObject;
                    hitTime = TestStage.GetTime();
                }

                private void OnLookEnd()
                {
                    float duration = TestStage.GetTime() - hitTime;
                    GetComponent<EventSender>().SetEventValue("ObjectName", lookingAt.name);
                    GetComponent<EventSender>().SetEventValue("Duration", duration);
                    GetComponent<EventSender>().Deploy();
                    lookingAt = null;
                }

                public GameObject getLookingAt()
                {
                    return lookingAt;
                }
            }
        }
    }
}
