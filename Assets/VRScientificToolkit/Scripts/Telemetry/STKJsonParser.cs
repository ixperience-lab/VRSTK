using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace STK
{
    ///<summary>Creates JSON strings from Testcontroller and event data.</summary>
    public static class STKJsonParser
    {

        private static string startString;
        private static string eventString;
        private static string endString;
        private static string[] stageString;
        private static int currentStage = 0;

        private static STKSettings settings = Resources.Load<STKSettings>("STKSettings");
        private static Hashtable latestStage;


        public static void TestStart(Hashtable p) //Called by TestController when a new Stages starts
        {
            if (stageString == null)
            {
                stageString = new string[STKTestController.numberOfStages];
            }
            int i = 0;
            eventString = "";
            startString = "{\n";
            startString += "\"TimeStarted\": \"" + System.DateTime.Now.Hour + ":" + System.DateTime.Now.Minute + ":" + System.DateTime.Now.Second + "\", \n";
            startString += "\"DateStarted\": \"" + System.DateTime.Now.Year + "." + System.DateTime.Now.Month + "." + System.DateTime.Now.Day + "\"";
            if (p.Count > 0) startString += ", \n";
            foreach (string s in p.Keys)
            {
                startString += "\"" + s + "\": " + FormatObject(TestStringFormat(p[s].ToString()));
                if (i < p.Keys.Count - 1)
                {
                    startString += ", \n";
                }
                i++;
            }
            latestStage = p; //Latest stage is saved in case another file is generated
        }

        public static void AddRunningProperties(Hashtable p)
        {
            int i = 0;
            if (latestStage != null && latestStage.Count > 0)
            {
                startString += ", \n";
            }
            foreach (string s in p.Keys)
            {
                startString += "\"" + s + "\": " + FormatObject(TestStringFormat(p[s].ToString()));
                if (i < p.Keys.Count - 1)
                {
                    startString += ", \n";
                }
                i++;
            }
        }

        public static void ReceiveEvents(Hashtable events)
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US"); //So it doesn't use commas for floats
            System.Text.StringBuilder sb = new System.Text.StringBuilder("");
            foreach (string s in events.Keys)
            {
                sb.Append(",\n");
                List<STKEvent> eventList = (List<STKEvent>)events[s];
                sb.Append("\"").Append(eventList[0].eventName).Append("\":\n[\n");
                int eventListIndex = 0;
                foreach (STKEvent e in eventList)
                {
                    sb.Append("{\n\"time\": ").Append(e.time).Append(",\n");
                    int objectsIndex = 0;
                    foreach (string o in e.objects.Keys)
                    {
                        sb.Append("\"").Append(o).Append("\": ").Append(FormatObject(e.objects[o])).Append("");
                        if (objectsIndex < e.objects.Keys.Count - 1)
                        {
                            sb.Append(",\n");
                        }
                        objectsIndex++;
                    }
                    sb.Append("\n}");
                    if (eventListIndex < eventList.Count - 1)
                    {
                        sb.Append(",\n");
                    }
                    eventListIndex++;
                }
                sb.Append("]\n");
            }
            eventString = sb.ToString();
        }

        private static System.Object TestStringFormat(string s) //Takes a String and tests if it can be converted to a float or int
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            int testInt;
            if (int.TryParse(s, out testInt))
            {
                return testInt;
            }
            float testFloat;
            if (float.TryParse(s, out testFloat))
            {
                float testChar;
                foreach (char c in s)
                {
                    if (c != '.' && !float.TryParse(c.ToString(), out testChar))
                    {
                        return s;
                    }
                }
                return testFloat;
            }
            return s;
        }

        private static string FormatObject(System.Object o) //Takes any variable, checks its type and returns a string that will format it correctly in JSON
        {
            if (o != null)
            {
                System.Text.StringBuilder sb = new System.Text.StringBuilder("");
                if (o.GetType() == typeof(string))
                {
                    sb.Append("\"").Append(o).Append("\"");
                    return sb.ToString();
                }
                else if (o.GetType() == typeof(int) || o.GetType() == typeof(float) || o.GetType() == typeof(double))
                {
                    return o.ToString();
                }
                else if (o.GetType() == typeof(bool))
                {
                    sb.Append("\"").Append(o.ToString()).Append("\"");
                    return sb.ToString();
                }
                else if (o.GetType() == typeof(Vector2))
                {
                    Vector2 v = (Vector2)o;
                    sb.Append("[").Append(v.x).Append(",").Append(v.y).Append("]");
                    return sb.ToString();
                }
                else if (o.GetType() == typeof(Vector3))
                {
                    Vector3 v = (Vector3)o;
                    sb.Append("[").Append(v.x).Append(",").Append(v.y).Append(",").Append(v.z).Append("]");
                    return sb.ToString();
                }
                else if (o.GetType() == typeof(Vector4))
                {
                    Vector4 v = (Vector4)o;
                    sb.Append("[").Append(v.x).Append(",").Append(v.y).Append(",").Append(v.z).Append(",").Append(v.w).Append("]");
                    return sb.ToString();
                }
                else if (o.GetType() == typeof(Quaternion))
                {
                    Quaternion v = (Quaternion)o;
                    sb.Append("[").Append(v.x).Append(",").Append(v.y).Append(",").Append(v.z).Append(",").Append(v.w).Append("]");
                    return sb.ToString();
                }
            }
            Debug.LogWarning("Formatting Object unsuccessful. Returning empty.");
            return "";
        }

        public static void TestEnd() //Called at the end of a stage
        {
            endString = "}\n";
            stageString[currentStage] = "\"Stage" + currentStage.ToString() + "\":" + startString + eventString + endString;
            currentStage++;
            if (currentStage >= stageString.Length)
            {
                Debug.Log("Creating final");
                CreateFile();
            }
        }

        public static void SaveRunning() //Saves an unfinished Experiment/Stage
        {
            ReceiveEvents(STKEventReceiver.GetEvents());
            STKEventReceiver.ClearEvents();
            endString = "}\n";
            stageString[currentStage] = "\"Stage" + currentStage.ToString() + "\":" + startString + eventString + endString;
            CreateFile();
            startString = null;
            stageString = null;
            eventString = null;
            endString = null;
            TestStart(latestStage);
        }

        public static string CreateFile() //Called at the end of the experiment. Completes JSON String and Saves it as a file
        {
            string fullString = "{\n";
            for (int i = 0; i < stageString.Length; i++)
            {
                fullString += stageString[i];
                if (i < stageString.Length - 1)
                {
                    if (stageString[i + 1] != null)
                    {
                        fullString += ",\n";
                    }
                }
            }
            fullString += "}";
            string path = (settings.jsonPath + "\\" + System.DateTime.Now.Month + "-" + System.DateTime.Now.Day + "_" + System.DateTime.Now.Hour + "-" + System.DateTime.Now.Minute + "-" + System.DateTime.Now.Second + ".json");
            using (StreamWriter sw = File.AppendText(path))
            {
                sw.Write(fullString);
            }
            currentStage = 0;
            return fullString;
        }
    }
}
