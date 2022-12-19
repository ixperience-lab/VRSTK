using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Text;
using SimpleJSON;
using UnityEngine;

namespace VRSTK
{
    namespace Scripts
    {
        [Serializable]
        public partial class JSONArray : JSONNode
        {
            [SerializeField]
            private List<JSONNode> m_List = new List<JSONNode>();

            public List<JSONNode> List
            {
                get { return m_List; }
                set { m_List = value; }
            }


            private bool inline = false;
            public override bool Inline
            {
                get { return inline; }
                set { inline = value; }
            }

            public void ConvertFromJSONArrayToVRSTKJSONArray(SimpleJSON.JSONArray input)
            {
                for (int i = 0; i < input.Count; i++)
                    m_List.Add(input[i]);
            }

            public override JSONNodeType Tag { get { return JSONNodeType.Array; } }
            public override bool IsArray { get { return true; } }
            public override Enumerator GetEnumerator() { return new Enumerator(m_List.GetEnumerator()); }

            public override JSONNode this[int aIndex]
            {
                get
                {
                    if (aIndex < 0 || aIndex >= m_List.Count)
                        return new JSONLazyCreator(this);
                    return m_List[aIndex];
                }
                set
                {
                    if (value == null)
                        value = JSONNull.CreateOrGet();
                    if (aIndex < 0 || aIndex >= m_List.Count)
                        m_List.Add(value);
                    else
                        m_List[aIndex] = value;
                }
            }

            public override JSONNode this[string aKey]
            {
                get { return new JSONLazyCreator(this); }
                set
                {
                    if (value == null)
                        value = JSONNull.CreateOrGet();
                    m_List.Add(value);
                }
            }

            public override int Count
            {
                get { return m_List.Count; }
            }

            public override void Add(string aKey, JSONNode aItem)
            {
                if (aItem == null)
                    aItem = JSONNull.CreateOrGet();
                m_List.Add(aItem);
            }

            public override JSONNode Remove(int aIndex)
            {
                if (aIndex < 0 || aIndex >= m_List.Count)
                    return null;
                JSONNode tmp = m_List[aIndex];
                m_List.RemoveAt(aIndex);
                return tmp;
            }

            public override JSONNode Remove(JSONNode aNode)
            {
                m_List.Remove(aNode);
                return aNode;
            }

            public override IEnumerable<JSONNode> Children
            {
                get
                {
                    foreach (JSONNode N in m_List)
                        yield return N;
                }
            }


            internal override void WriteToStringBuilder(StringBuilder aSB, int aIndent, int aIndentInc, JSONTextMode aMode)
            {
                aSB.Append('[');
                int count = m_List.Count;
                if (inline)
                    aMode = JSONTextMode.Compact;
                for (int i = 0; i < count; i++)
                {
                    if (i > 0)
                        aSB.Append(',');
                    if (aMode == JSONTextMode.Indent)
                        aSB.AppendLine();

                    if (aMode == JSONTextMode.Indent)
                        aSB.Append(' ', aIndent + aIndentInc);
                    m_List[i].WriteToStringBuilder(aSB, aIndent + aIndentInc, aIndentInc, aMode);
                }
                if (aMode == JSONTextMode.Indent)
                    aSB.AppendLine().Append(' ', aIndent);
                aSB.Append(']');
            }
        }
    }
}
