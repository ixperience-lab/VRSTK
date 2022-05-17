using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class PageParameters : MonoBehaviour
            {
                //Completion Times
                /// <summary>
                /// The variables TIME001 etc. store the time (in seconds) that a participant stayed on a questionnaire page.
                /// </summary>
                [SerializeField]
                private float _TIME_nnn = 0f;

                public float TIME_nnn
                {
                    get { return _TIME_nnn; }
                    set { _TIME_nnn = value; }
                }

                [SerializeField]
                private float _current_Time_nnn = 0f;

                private float _started_time = 0f;

                // Start is called before the first frame update
                void Start()
                {
                    _started_time = TestStage.GetTime();
                }

                // Update is called once per frame
                void Update()
                {
                    _current_Time_nnn = TestStage.GetTime() - _started_time;
                    TIME_nnn += _current_Time_nnn;
                }
            }
        }
    }
}
