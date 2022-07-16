using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class ActivateModels : MonoBehaviour
{
    [SerializeField]
    public List<GameObject> modelList = new List<GameObject>();
    
    [SerializeField]
    public GameObject _currentActivatedModel;

    [SerializeField]
    public int _currenSelectedtIndex = 0;

    [SerializeField]
    public string _currentActivatedModelName;

    //private bool delay = true;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void FirstPage()
    {
        _currenSelectedtIndex = 0;
        _currentActivatedModel = modelList[_currenSelectedtIndex];
        _currentActivatedModel.SetActive(true);
        _currentActivatedModelName = _currentActivatedModel.name;
    }

    public void LastPage()
    {
        modelList[_currenSelectedtIndex].SetActive(false);
    }

    public void StartPage()
    {
        modelList[_currenSelectedtIndex].SetActive(false);
    }

    public void GoToNextPage()
    {
        if (_currenSelectedtIndex + 1 >= modelList.Capacity)
        {
            Debug.Log("Test done. \n Reset now with 'R'");
            return;
        }

        modelList[_currenSelectedtIndex].SetActive(false);
        _currenSelectedtIndex++;
        modelList[_currenSelectedtIndex].SetActive(true);

        _currentActivatedModel = modelList[_currenSelectedtIndex];
        _currentActivatedModelName = _currentActivatedModel.name;
    }

    public void GoToPreviousPage()
    {
        if (_currenSelectedtIndex == 0)
        {
            Debug.Log("Already at first element.");
            return;
        }
        modelList[_currenSelectedtIndex].SetActive(false);
        _currenSelectedtIndex--;
        modelList[_currenSelectedtIndex].SetActive(true);

        _currentActivatedModel = modelList[_currenSelectedtIndex];
        _currentActivatedModelName = _currentActivatedModel.name;
    }

    public void ResetModels()
    {
        foreach (GameObject model in modelList)
        {
            model.SetActive(false);
        }
        _currenSelectedtIndex = 0;

        modelList.Shuffle();
        string arrangement = "";
        foreach (GameObject model in modelList)
        {
            arrangement += model.name + ",";
        }

        Debug.Log("Reihenfolge: \n" + arrangement);
        Debug.Log("Test reset");

        _currentActivatedModel = modelList[_currenSelectedtIndex];
        _currentActivatedModelName = _currentActivatedModel.name;
    }

    public void StartActivateModels()
    {
        modelList[_currenSelectedtIndex].SetActive(true);
        _currentActivatedModel = modelList[_currenSelectedtIndex];
        _currentActivatedModelName = _currentActivatedModel.name;
        Debug.Log("Test started.");
    }
}

public static class ThreadSafeRandom
{
    [ThreadStatic] private static System.Random local;

    public static System.Random ThisThreadsRandom
    {
        get { return local ?? (local = new System.Random(unchecked(Environment.TickCount * 31 + Thread.CurrentThread.ManagedThreadId))); }
    }
}

static class ListShuffleExtension
{
    public static void Shuffle<T>(this IList<T> list)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = ThreadSafeRandom.ThisThreadsRandom.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }
}
