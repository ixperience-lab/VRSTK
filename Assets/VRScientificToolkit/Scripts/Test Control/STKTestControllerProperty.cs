using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace STK
{
    ///<summary>A property is an input the user can make for a test stage, for example a name or the answer to a question.</summary>
    public class STKTestControllerProperty : MonoBehaviour
    {

        public Text text;
        public InputField inputField;
        public Toggle toggle;
        private string value;
        // Use this for initialization
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            if (inputField != null)
            {
                value = inputField.text;
            }
            else if (toggle != null)
            {
                value = toggle.isOn.ToString();
            }
        }

        public string GetValue()
        {
            return value;
        }

        public void Clear()
        {
            value = null;
            if (inputField != null)
            {
                inputField.text = null;
            }
        }

    }
}
