namespace STK
{
    /// <summary>
    /// Contain configuration of a specific App.
    /// </summary>
    static class STKCortexAppConfig
    {
        public static string AppUrl              = "wss://localhost:6868";
        public static string AppName             = "vrstk-app-epocx";
    
        /// <summary>
        /// Name of directory where contain tmp data and logs file.
        /// </summary>
        public static string TmpAppDataDir       = "vrstk-app-epocx";
        public static string ClientId            = "tzZ4CeRQ6y7Z97daz3J28ArIVBfFixwN2SU5eJaY";
        public static string ClientSecret        = "7QZLPApdlO0aqWusSd7ooTaLA8mKMOFWFsHliCkrP3LmK3uSe6aDGSLsRMPvk56PzUe1jFUNuIVc3FlCo7X4Wy703Q0dyy0LG2YITMRvtPsChIOanObKKwD8stxbcrjY";
        public static string AppVersion          = "1.0.0";
    
        /// <summary>
        /// License Id is used for App
        /// In most cases, you don't need to specify the license id. Cortex will find the appropriate license based on the client id
        /// </summary>
        public static string AppLicenseId        = "";
    }
}