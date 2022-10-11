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
            //if (p0.z < -7.0f) p0.z = -6.99f;
            //if (p1.z < -7.0f) p1.z = -6.99f;

            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            lineRenderer.positionCount += 2;
            lineRenderer.SetPosition(lineRenderer.positionCount - 2, p0);
            lineRenderer.SetPosition(lineRenderer.positionCount - 1, p1);
            _lastSaccdesPositions = SaccadesPositions;

            //if (!_usedStartPositionOnce) //(((p0.z == -2.5f) || (p1.z == -2.5f)) && !_usedStartPositionOnce)
            //{
            //    // Correction of start position stage 1
            //    if (p0.z > -2.0f) { p0.x = 0f; p0.y = 0f; p0.z = -2.5f; }
            //    if (p1.z > -2.0f) { p1.x = 0f; p1.y = 0f; p1.z = -2.5f; }

            //    lineRenderer.positionCount += 2;
            //    lineRenderer.SetPosition(lineRenderer.positionCount - 2, p0);
            //    lineRenderer.SetPosition(lineRenderer.positionCount - 1, p1);
            //    _lastSaccdesPositions = SaccadesPositions;
            //    _usedStartPositionOnce = true;
            //}

            //if (_usedStartPositionOnce ) //((((p0.z < -2.5f) || (p1.z < -2.5f)) && _usedStartPositionOnce))
            //{
            //    if ((p0.z < -5.9f) && (p0.z > -6.6f))
            //    {
            //        p0.x -= 0.8f; 
            //        p0.z = -5.2f;
            //    }
            //    if ((p1.z < -5.9f) && (p1.z > -6.6f))
            //    {
            //        p1.x -= 0.8f; 
            //        p1.z = -5.2f;
            //        //float dist = Vector3.Distance(_CenterOfView.transform.position, p1);
            //        //Vector3 dir = _CenterOfView.transform.position - p1;
            //        //// p1 = _CenterOfView.transform.position + dir * dist;
            //        //p1 -= (dir * dist);
            //    }

            //    // lineRenderer.positionCount += 2;
            //    if (p0.z < -2.0f)
            //    {
            //        lineRenderer.positionCount += 1;
            //        lineRenderer.SetPosition(lineRenderer.positionCount - 1, p0);
            //    }
            //    if (p1.z < -2.0f)
            //    {
            //        lineRenderer.positionCount += 1;
            //        lineRenderer.SetPosition(lineRenderer.positionCount - 1, p1);
            //    }
            //    _lastSaccdesPositions = SaccadesPositions;
            //}
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
                    //// Correction of z-values while theres no raycast on page
                    //if (position.z < -7.0f) position.z = -6.99f;
                    //// Correction of start position 
                    //if (position.z > -2.0f) continue;
                    //// Correction of centerofview
                    //if ((position.z < -5.9f) && (position.z > -6.6f))
                    //{
                    //    position.x -= 0.8f; 
                    //    position.z = -5.2f;
                    //}

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
