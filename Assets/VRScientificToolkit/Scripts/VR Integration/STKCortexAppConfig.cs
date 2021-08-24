namespace STK
{
    /// <summary>
    /// Contain configuration of a specific App.
    /// </summary>
    static class STKCortexAppConfig
    {
        public static string AppUrl              = "wss://localhost:6868";
        public static string AppName             = "UnityApp";
    
        /// <summary>
        /// Name of directory where contain tmp data and logs file.
        /// </summary>
        public static string TmpAppDataDir       = "UnityApp";
        public static string ClientId            = "oh9TRvjlzEkDl0cVoMwphQ5C3IRHIgZX6tmJci7r";
        public static string ClientSecret        = "Mh93oZJPuzGFIAlUEzKUtJYyLcTKflOI4h0FuplFmk8OsHD0ospRUTAYYnCJr8ByogJooESsYq7cz9lrrlpKZpLXS6VnXJiNJBICWhSNIoQpDnZ23K9ScZJhCWiHmAcX";
        public static string AppVersion          = "1.0.0";
    
        /// <summary>
        /// License Id is used for App
        /// In most cases, you don't need to specify the license id. Cortex will find the appropriate license based on the client id
        /// </summary>
        public static string AppLicenseId        = "";
    }
}