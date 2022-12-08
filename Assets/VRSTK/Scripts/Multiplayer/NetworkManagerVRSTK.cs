using Mirror;
using UnityEditor;
using UnityEngine;

namespace VRSTK.Scripts.Multiplayer
{
    /// <summary>
    /// Manages connections and initializes player trackers
    /// </summary>
    public class NetworkManagerVRSTK : NetworkManager
    {
        [SerializeField] private CopyTransform player1TrackerHead;
        [SerializeField] private CopyTransform player1TrackerLeft;
        [SerializeField] private CopyTransform player1TrackerRight;
        [SerializeField] private CopyTransform player2TrackerHead;
        [SerializeField] private CopyTransform player2TrackerLeft;
        [SerializeField] private CopyTransform player2TrackerRight;
        
        
        public override void OnStartServer()
        {
            base.OnStartServer();
            Debug.Log("Server Started!");
        }

        public override void OnServerAddPlayer(NetworkConnectionToClient conn)
        {
            Transform startPos = GetStartPosition();
            GameObject player = startPos != null
                ? Instantiate(playerPrefab, startPos.position, startPos.rotation)
                : Instantiate(playerPrefab);

            // instantiating a "Player" prefab gives it the name "Player(clone)"
            // => appending the connectionId is WAY more useful for debugging!
            player.name = $"{playerPrefab.name} [connId={conn.connectionId}]";
            NetworkServer.AddPlayerForConnection(conn, player);

            if (!player1TrackerHead.origin)
            {
                Debug.Log("Player 1 (Host Mode) spawned!");
                player1TrackerHead.origin = player.GetComponent<Player>().netHead.transform.GetChild(0).transform;
                player1TrackerLeft.origin = player.GetComponent<Player>().netLeft.transform.GetChild(0).transform;
                player1TrackerRight.origin = player.GetComponent<Player>().netRight.transform.GetChild(0).transform;
            }
            else
            {
                Debug.Log("Player 2 spawned!");
                player2TrackerHead.origin = player.GetComponent<Player>().netHead.transform.GetChild(0).transform;
                player2TrackerLeft.origin = player.GetComponent<Player>().netLeft.transform.GetChild(0).transform;
                player2TrackerRight.origin = player.GetComponent<Player>().netRight.transform.GetChild(0).transform;
            }


        }
    }
}
