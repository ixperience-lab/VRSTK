using UnityEngine;

namespace VRSTK.Scripts.Multiplayer
{
    public class CopyTransform : MonoBehaviour
    {
        [SerializeField] public Transform origin;
        [SerializeField] private Transform target;

        private void Update()
        {
            if (!origin) return;
            target.position = origin.position;
            target.rotation = origin.rotation;
        }
    }
}
