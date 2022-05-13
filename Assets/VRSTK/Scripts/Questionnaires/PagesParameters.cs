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
            public class PagesParameters : MonoBehaviour
            {
                // Interview Identification
                /// <summary>
                /// Time when the participant started the interview.
                /// </summary>
                [SerializeField]
                private string _STARTED = "";

                public string STARTED
                {
                    get { return _STARTED; }
                    set { _STARTED = value; }
                }

                // Interview Progress variables
                /// <summary>
                /// Time when the participant most recently clicked the “next”
                /// </summary>
                [SerializeField]
                private string _LASTDATA = "";

                public string LASTDATA
                {
                    get { return _LASTDATA; }
                    set { _LASTDATA = value; }
                }

                /// <summary>
                /// The page most recently answered (and sent via Next) by the participant
                /// </summary>
                [SerializeField]
                private int _LASTPAGE = 0;
                public int LASTPAGE
                {
                    get { return _LASTPAGE; }
                    set { _LASTPAGE = value; }
                }

                /// <summary>
                /// The greatest number of any page answered by the participant.
                /// </summary>
                [SerializeField]
                private int _MAXPAGE = 0;

                public int MAXPAGE
                {
                    get { return _MAXPAGE; }
                    set { _MAXPAGE = value; }
                }

                /// <summary>
                ///  Did the participant reach the goodbye page (1) or not (0).
                /// </summary>
                [SerializeField]
                private int _FINISHED = 0;
                public int FINISHED
                {
                    get { return _FINISHED; }
                    set { _FINISHED = value; }
                }


                // Completion Times
                /// <summary>
                /// An index that indicates how much faster a participant has completed the questionnaire than the typical participant (median) has done.
                /// </summary>
                [SerializeField]
                private int _TIME_RSI = 0;

                public int TIME_RSI
                {
                    get { return _TIME_RSI; }
                    set { _TIME_RSI = value; }
                }

                // Start is called before the first frame update
                void Start()
                {
                    if (TestStage.GetStarted())
                    {
                        _STARTED = System.DateTime.Now.ToString();
                    }
                }

                // Update is called once per frame
                void Update()
                {
                    if (TestStage.GetStarted())
                    {

                    }
                }
            }
        }
    }
}
