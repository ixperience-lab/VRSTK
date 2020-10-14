using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace STK
{
    ///<summary>Defines Variable Types which can be serialized in events</summary>
    public static class STKEventTypeChecker
    {

        public static System.Type[] allowedTypes = { typeof(int), typeof(float), typeof(string), typeof(bool), typeof(Vector2), typeof(Vector3), typeof(Vector4), typeof(Quaternion), typeof(System.String), typeof(UnityEngine.Vector2), typeof(UnityEngine.Vector3), typeof(UnityEngine.Vector4) };

        public static bool IsValid(System.Type typeToTest)
        {
            foreach (System.Type t in allowedTypes)
            {
                if (typeToTest == t)
                {
                    return true;
                }
            }
            return false;
        }

        public static int getIndex(System.Type typeToTest)
        {
            Debug.Log(typeToTest);
            for (int i = 0; i < allowedTypes.Length; i++)
            {
                if (typeToTest == allowedTypes[i])
                {
                    return i;
                }
            }
            return 0;
        }
    }
}
