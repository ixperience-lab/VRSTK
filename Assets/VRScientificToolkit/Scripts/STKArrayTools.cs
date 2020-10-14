using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace STK
{
    ///<summary>This class is used to add elements to an array and clean null references from an array</summary>
    public static class STKArrayTools
    {

        public static object[] AddElement(object o, object[] array)
        {
            if (array == null)
            {
                object[] createdArray = new object[1];
                createdArray[0] = o;
                return createdArray;
            }
            object[] newArray = new object[array.Length + 1];

            for (int i = 0; i < array.Length; i++)
            {
                newArray[i] = array[i];
            }

            newArray[newArray.Length - 1] = o;
            return newArray;
        }

        public static object[] ClearNullReferences(object[] array)
        {
            int numberOfObjects = 0;
            foreach (object o in array)
            {
                if (o != null)
                {
                    numberOfObjects++;
                }
            }
            object[] newArray = new object[numberOfObjects];
            int currentindex = 0;

            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] != null)
                {
                    newArray[currentindex] = array[i];
                    currentindex++;
                }
            }
            return newArray;
        }
    }
}
