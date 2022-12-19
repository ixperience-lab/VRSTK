using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            [Serializable]
            public class QuestionsAnswerRecord
            {
                [SerializeField]
                private string _question;

                public string Question
                {
                    get { return _question; }
                    set { _question = value; }
                }

                [SerializeField]
                private int _questionedCounter;

                public int QuestionedCounter
                {
                    get { return _questionedCounter; }
                    set { _questionedCounter = value; }
                }


                [SerializeField]
                private int _answeredCounter;

                public int AnsweredCounter
                {
                    get { return _answeredCounter; }
                    set { _answeredCounter = value; }
                }

                public QuestionsAnswerRecord(string question, int questionedCounter, int answer)
                {
                    _question = question;
                    _questionedCounter = questionedCounter;
                    _answeredCounter = answer;
                }

                public float CalculateWeightFactor()
                {
                    return _questionedCounter > 0 ? (float)((float)_answeredCounter / (float)_questionedCounter) : 0f ;
                }
            }
        }
    }
}
