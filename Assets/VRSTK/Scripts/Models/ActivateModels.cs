using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class ActivateModels : MonoBehaviour
{
    public List<GameObject> modelList = new List<GameObject>();
    public GameObject _currentActivatedModel;
    private int currentIndex = 0;
    
    private bool delay = true;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void GoToNextPage()
    {
        if (currentIndex + 1 >= modelList.Capacity)
        {
            Debug.Log("Test done. \n Reset now with 'R'");
            return;
        }
        else if (delay)
        {
            delay = !delay;
            modelList[currentIndex].SetActive(false);
            return;
        }
        delay = true;
        modelList[currentIndex].SetActive(false);
        currentIndex++;
        modelList[currentIndex].SetActive(true);

        _currentActivatedModel = modelList[currentIndex];
    }

    public void GoToPreviousPage()
    {
        if (currentIndex == 0)
        {
            Debug.Log("Already at first element.");
            return;
        }
        modelList[currentIndex].SetActive(false);
        currentIndex--;
        modelList[currentIndex].SetActive(true);

        _currentActivatedModel = modelList[currentIndex];
    }

    public void ResetModels()
    {
        foreach (GameObject model in modelList)
        {
            model.SetActive(false);
        }
        currentIndex = 0;

        modelList.Shuffle();
        string arrangement = "";
        foreach (GameObject model in modelList)
        {
            arrangement += model.name + ",";
        }

        Debug.Log("Reihenfolge: \n" + arrangement);
        Debug.Log("Test reset");

        _currentActivatedModel = modelList[currentIndex];
    }

    public void StartActivateModels()
    {
        modelList[currentIndex].SetActive(true);
        _currentActivatedModel = modelList[currentIndex];
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
