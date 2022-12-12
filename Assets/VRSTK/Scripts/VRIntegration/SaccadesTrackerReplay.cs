using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SaccadesTrackerReplay : MonoBehaviour
{
    [SerializeField]
    private string _saccadesPositions;

    private string _lastSaccdesPositions = "";

    public string SaccadesPositions
    {
        get
        {
            return _saccadesPositions;
        }
        set
        {
            _saccadesPositions = value;
            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            lineRenderer.enabled = true;
            SaccadesReplay();
        }
    }

    [SerializeField]
    private string _fixationPosition;

    public string FixationPosition
    {
        get
        {
            return _fixationPosition;
        }
        set
        {
            _fixationPosition = value;
            FixationReplay();
        }
    }

    [SerializeField]
    private bool _usedStartPositionOnce = false;

    [SerializeField]
    public GameObject _CenterOfView;

    public Material _material;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void SaccadesReplay()
    {
        if (SaccadesPositions != string.Empty && SaccadesPositions != _lastSaccdesPositions)
        {
            string[] temp = SaccadesPositions.Split(';');
            string[] pArray = temp[0].Split(',');
            Vector3 p0 = new Vector3(float.Parse(pArray[0].Substring(1, pArray[0].Length - 1).Trim()), float.Parse(pArray[1].Trim()), float.Parse(pArray[2].Substring(0, pArray[2].Length - 1).Trim()));
            pArray = temp[1].Split(',');
            Vector3 p1 = new Vector3(float.Parse(pArray[0].Substring(1, pArray[0].Length - 1).Trim()), float.Parse(pArray[1].Trim()), float.Parse(pArray[2].Substring(0, pArray[2].Length - 1).Trim()));

            // Correction of z-values while theres no raycast on page stage 1
            //if (p0.z <= -7.0f) p0 *= (6.99f / Mathf.Abs(p0[2]));
            //if (p1.z <= -7.0f) p1 *= (6.99f / Mathf.Abs(p1[2]));

            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            lineRenderer.positionCount += 2;
            lineRenderer.SetPosition(lineRenderer.positionCount - 2, p0);
            lineRenderer.SetPosition(lineRenderer.positionCount - 1, p1);
            _lastSaccdesPositions = SaccadesPositions;
        }
    }

    private void FixationReplay()
    {
        if (FixationPosition != string.Empty)
        {
            string[] temp = FixationPosition.Split(';');
            for(int i = 0; i < temp.Length-1; i++)
            {
                string[] pArray = temp[i].Split(',');
                Vector3 position = new Vector3(float.Parse(pArray[0].Substring(1, pArray[0].Length - 1).Trim()), float.Parse(pArray[1].Trim()), float.Parse(pArray[2].Substring(0, pArray[2].Length - 1).Trim()));
                string objectName = "ReplaySphere_" + position.x.ToString().Replace(".","_") + position.y.ToString().Replace(".", "_") + position.z.ToString().Replace(".", "_");
                string tagName = "ReplaySphere_";
                Debug.Log(objectName);
                {
                    // Correction of z-values while theres no raycast on page
                    //if (position.z <= -7.0f) position = (6.99f / Mathf.Abs(position[2])) * position;
                    // Correction of start position 
                    //if (position.z > -2.0f) continue;
                    
                    GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    sphere.tag = tagName;
                    sphere.name = objectName;
                    sphere.transform.position = new Vector3(position.x, position.y, position.z);
                    sphere.transform.localScale = new Vector3(0.03f, 0.03f, 0.03f);
                    sphere.GetComponent<MeshRenderer>().receiveShadows = false;
                    sphere.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    Renderer rend = sphere.GetComponent<Renderer>();
                    rend.material = _material;
                }
            }
            
        }
    }
}
