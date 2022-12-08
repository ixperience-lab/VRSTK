using System;
using Mirror;
using UnityEditor;
using UnityEngine;

namespace VRSTK.Scripts.Multiplayer
{
    /// <summary>
    /// Starts Host Mode (Server + local Client) if played from Editor
    /// </summary>
    [RequireComponent(typeof(NetworkManager))]
    public class NetworkManagerStartHost : MonoBehaviour
    {
        NetworkManager manager;

#if UNITY_EDITOR
        void Awake()
        {
            manager = GetComponent<NetworkManager>();
        }

        private void Start()
        {
            manager.StartHost();
        }
#endif
    }
    

}