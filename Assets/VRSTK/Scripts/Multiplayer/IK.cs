using UnityEngine;

namespace VRSTK.Scripts.Multiplayer
{
    public class IK : MonoBehaviour
    {
        public Transform head;
        public Transform right;
        public Transform left;
        
        public Transform netHead;
        public Transform netRight;
        public Transform netLeft;
        
        private void Update()
        {
            head.position = netHead.position;
            right.position = netRight.position;
            left.position = netLeft.position;
   
            head.rotation = netHead.rotation;
            right.rotation = netRight.rotation;
            left.rotation = netLeft.rotation;
        }
    }
}