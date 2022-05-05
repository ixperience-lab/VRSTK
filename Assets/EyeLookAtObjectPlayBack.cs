using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EyeLookAtObjectPlayBack : MonoBehaviour
{
    
    public Vector3 _eyeHitPoint;

    public string ObjectName;

    public System.Single Duration;

    public float Time;

    public Material _yellow;
    public Material _red;

    public Vector3 EyeHitPoint
    {
        get { return _eyeHitPoint; }
        set 
        {
             _eyeHitPoint = value;
            CreateGameObject();
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void CreateGameObject()
    {
        if (!ObjectName.Equals("CameraTracking") && !ObjectName.Equals("XROriginEye"))
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = ObjectName + "_" + Time;
            sphere.transform.position = new Vector3(_eyeHitPoint.x, _eyeHitPoint.y, _eyeHitPoint.z);
            sphere.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f);
            sphere.GetComponent<MeshRenderer>().receiveShadows = false;
            sphere.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            Renderer rend = sphere.GetComponent<Renderer>();

            if (Duration > 0.5f)
                rend.material = _red;
            else
                rend.material = _yellow;

            sphere.transform.parent = transform;
        }        
    }
}
