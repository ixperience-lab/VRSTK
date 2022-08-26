using Mirror;
using UnityEngine;

namespace VRSTK.Scripts.Multiplayer
{
    public class Player : NetworkBehaviour
    {
        [SerializeField] private GameObject netHead;
        [SerializeField] private GameObject netLeft;
        [SerializeField] private GameObject netRight;
        
        private GameObject _localHead;
        private GameObject _localLeft;
        private GameObject _localRight;

        private GameObject _theLocalPlayer;
        
        private void Update()
        {
            if (!isLocalPlayer) return; //isServer ||
            UpdateHeadAndHands();
        }
        
        public void Start()
        {
            //if (isServer) return;
            if (!isLocalPlayer) return;

            _theLocalPlayer = GameObject.Find("XROriginLocomotion");

            Transform cameraOffset = _theLocalPlayer.transform.Find("Camera Offset");

            _localHead = cameraOffset.Find("Main Camera").gameObject;
            _localLeft = cameraOffset.Find("LeftHand Controller").gameObject;//cameraOffset.Find("LeftHandController/LeftHandControllerDirect").gameObject;
            _localRight = cameraOffset.Find("RightHand Controller").gameObject;//cameraOffset.Find("RightHandController/RightHandControllerDirect").gameObject;

        }

        private void UpdateHeadAndHands()
        {
            // We are the local player.
            // We copy the values from the Rig's HMD and hand positions so they can be used for local positioning
            
            netHead.transform.position = _localHead.transform.position;
            netLeft.transform.position = _localLeft.transform.position;
            netRight.transform.position = _localRight.transform.position;
            
            netHead.transform.rotation = _localHead.gameObject.transform.rotation;
            netLeft.transform.rotation = _localLeft.transform.rotation;
            netRight.transform.rotation = _localRight.transform.rotation;
        }
        
    }
}