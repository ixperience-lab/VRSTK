using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace STK
{
    ///<summary>Detects which objects users are looking at and sends out events with how long they looked at it.</summary>
    public class STKLookDirection : MonoBehaviour
    {

        public STKEvent lookEvent;
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
            hitTime = STKTestStage.GetTime();
        }

        private void OnLookEnd()
        {
            float duration = STKTestStage.GetTime() - hitTime;
            GetComponent<STKEventSender>().SetEventValue("ObjectName", lookingAt.name);
            GetComponent<STKEventSender>().SetEventValue("Duration", duration);
            GetComponent<STKEventSender>().Deploy();
            lookingAt = null;
        }

        public GameObject getLookingAt()
        {
            return lookingAt;
        }
    }
}
