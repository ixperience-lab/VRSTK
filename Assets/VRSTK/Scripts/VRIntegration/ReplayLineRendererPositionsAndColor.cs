using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

public class ReplayLineRendererPositionsAndColor : MonoBehaviour
{

    [SerializeField]
    private string _positionsAndColorMessage;

    public string PositionsAndColorMessage
    {
        get
        {
            return _positionsAndColorMessage;
        }
        set
        {
            _positionsAndColorMessage = value;
            Replay();
        }
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (TestStage.GetStarted())
        {
            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            Color color = lineRenderer.startColor;
            Vector3[] positions = new Vector3[2];
            lineRenderer.GetPositions(positions);

            if (positions.Length == 2)
            {
                string message = string.Format("{0};{1};{2};{3};{4};{5};", positions[0].ToString(), positions[1].ToString(), color.r, color.g, color.b, color.a); 
                GetComponents<EventSender>()[2].SetEventValue("PositionsAndColorMessage_ReplayLineRendererPositionsAndColor", message);
                GetComponents<EventSender>()[2].Deploy();
            }
        }
    }

    private void Replay()
    {
        if (PositionsAndColorMessage != string.Empty)
        {
            string[] temp = PositionsAndColorMessage.Split(';');
            string[] pArray = temp[0].Split(',');
            Vector3 p0 = new Vector3(float.Parse(pArray[0].Substring(1, pArray[0].Length - 1).Trim()), float.Parse(pArray[1].Trim()), float.Parse(pArray[2].Substring(0, pArray[2].Length - 1).Trim()));
            pArray = temp[1].Split(',');
            Vector3 p1 = new Vector3(float.Parse(pArray[0].Substring(1, pArray[0].Length - 1).Trim()), float.Parse(pArray[1].Trim()), float.Parse(pArray[2].Substring(0, pArray[2].Length - 1).Trim()));

            Color color = new Color(float.Parse(temp[2]), float.Parse(temp[3]), float.Parse(temp[4]), float.Parse(temp[5]));

            //XRInteractorLineVisual xrInteractorLineVisual = GetComponent<XRInteractorLineVisual>();
            //xrInteractorLineVisual.enabled = true;

            LineRenderer lineRenderer = GetComponent<LineRenderer>();
            //lineRenderer.enabled = true;

            lineRenderer.SetColors(color, color);
            lineRenderer.positionCount = 2;
            lineRenderer.SetPosition(0, p0);
            lineRenderer.SetPosition(1, p1);
        }
    }
}
