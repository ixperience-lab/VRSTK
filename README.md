# Virtual Reality Scientific Toolkit (VRSTK)

The Virtual Reality Scientific Toolkit facilitates the creation and execution of experiments in VR environments by making object tracking and data collection easier.

## Quickstart Anleitung
Das VR Scientific Toolkit vereinfacht die Durchführung von Experimenten in VRUmgebungen in Unity. Es ermöglicht unter anderem die Aufzeichnung von verschiedenen Forschungsdaten, sowie deren Speicherung in einer JSON-Datei. In dieser Quickstart-Anleitung wird erklärt, wie Sie das Toolkit in Ihr Projekt einbinden und die wichtigsten Funktionen nutzen können.

### Installation
Stellen Sie vor der Installation des Toolkits sicher, dass das SteamVR-Plugin installiert ist und setzen Sie in den Player Settings die Variable „Scripting Runtime Version“ auf „.NET 4.x Equivalent“.
Anschließend importieren Sie das Unity Package, indem Sie im Asset-Browser „Import Package“ wählen und hier das Toolkit auswählen. Sie finden in Ihrem Projekt nun den „VrScientificToolkit“ Ordner, in welchem sich Skripte und Prefabs befinden, die Sie für verschiedene Funktionen nutzen können. Für eine Demonstration der Fähigkeiten des Toolkits können Sie die „ExampleScene“ betrachten, in welcher sich eine beispielhafte Anwendung findet.

### Festlegen der Einstellungen
Nachdem Sie das Plugin installiert haben, sollten Sie zunächst die Einstellungen des Toolkits an Ihre Bedürfnisse anpassen. Die Einstellungen sind im Resources-Ordner unter dem Namen „STKSettings“ zu finden. Hier können Sie verschiedene Einstellungen, wie den Speicherort für die generierten JSON-Dateien, auswählen. Viele Interface Elemente sind mit Tooltips ausgestattet, die weitere Informationen zu
deren Funktion bieten. Hierzu können Sie einfach die Maus über das entsprechende Element halten.

### Testcontroller
Der Testcontroller dient der Steuerung des Experiments. Über ihn lassen sich während des Experiments Daten eingeben und Funktionen ausführen. Er ist für den Probanden in VR nicht sichtbar, sondern wird nur dem Testleiter auf dem Bildschirm angezeigt. Um den Testcontroller zu erstellen, ziehen Sie ihn aus dem Prefabs-Menü in die Szene. Klicken Sie anschließend im Inspektor auf „Add Stage“, um einen ersten Experimentabschnitt hinzuzufügen. Über Abschnitte bzw. Stages definiert sich der Aufbau des Experiments, auch in der automatisch generierten JSON-Datei. Für diesen Abschnitt können Sie, erneut über den Inspektor, Properties hinzufügen. Dabei handelt es sich um Eingabefelder, die zum Zeitpunkt des Experiments dazu genutzt werden können, Daten, wie zum Beispiel den Namen eines Probanden, einzugeben. Außerdem können im Inspektor Buttons erstellt werden, welche beliebige Funktionen ausführen können.

![Testcontrollers](https://user-images.githubusercontent.com/59827885/125065922-b3300000-e0b2-11eb-97b1-a7bbde06b5f7.jpg)
*Beispiel eines Testcontrollers*

### Tracking
Über die Tracking Funktion können beliebige Variablen von Objekten aufgezeichnet und in einer JSON-Datei gespeichert werden. Bevor Sie die Tracking-Funktion verwenden, ziehen Sie bitte das „STKTrackedObjects“ Prefab in die Szene. Um Variablen eines Objektes aufzuzeichnen, öffnen Sie zunächst das Tracking-Interface
über Window/VR Scientific Toolkit/Track Object. Das Tracking Interface zeigt immer die Komponenten des aktuell in der Szene angewählten Objektes an. Wird eine Komponente ausgewählt, können zu dieser Komponente gehörende, öffentliche Variablen ausgewählt werden. 

Nach einem Klick auf „Create Tracker“ erhält das Objekt ein Skript, welches die gewählten Variablen aufzeichnet. Im Inspektor kann ein Intervall eingestellt werden, in dem die Aufzeichnung erfolgt. Damit das Event im definierten Intervall abgesendet wird, setzen Sie außerdem „Timed Interval“ auf True. Das Event kann ebenfalls manuell über einen Aufruf der Deploy-Funktion des Eventsender Skriptes abgesendet werden.

![Tracking](https://user-images.githubusercontent.com/59827885/125066086-e5416200-e0b2-11eb-8b82-6afd14fed09c.png)
*Das Tracking Interface*

### Definition eigener Events
Neben automatisch generierten Tracker-Events können Sie auch eigene Events erstellen, welche zum Beispiel ausgelöst werden können, wenn ein bestimmter Gegenstand in der Szene aufgehoben wird. Diese Events können beliebige Attribute enthalten. Ein eigenes Event können Sie über das Kontextmenü im „Project“ Fenster von Unity über die Punkte Create/VR Scientific Toolkit/STKEvent definieren. Wählen sie dieses Event aus, um Parameter zu definieren. Bei jedem Parameter können Sie den Namen sowie einen Datentyp aus einer Liste von kompatiblen Datentypen auswählen.

Um dieses Event zu versenden, wird eine „STKEventSender“ Komponente auf einem GameObject benötigt. Legen Sie hier das erstellte Event als „eventBase“ fest. Um einem Eventattribut während des Testablaufs einen Wert zuzuteilen, können Sie „SetEventValue“ aufrufen. Die Funktion wird anschließend über „Deploy“ dem
Eventreceiver zur Speicherung übergeben. 

![Img__3_](https://user-images.githubusercontent.com/59827885/125066124-eecaca00-e0b2-11eb-9c0a-741136a22dc5.png)
*Definition eines Events*

### Wiedergabe eines Experimentes
Es ist möglich, den Ablauf eines Experimentes wiederherzustellen, um ihn erneut in der Unity Szene betrachten zu können. Hierfür werden die Daten von Tracker-Events genutzt, um sämtliche getrackten Variablen wieder auf den Wert eines bestimmten Zeitpunktes zu setzen. Um das Wiedergabe-Interface zu öffnen, wählen Sie den Menüpunkt Window/VR Scientific Toolkit/JSON Playback. Anschließend deaktivieren
Sie die VR-Unterstützung in den Player-Einstellungen von Unity, damit keine Konflikte mit den SteamVR-Treibern entstehen. Wenn das Playback-Fenster geöffnet ist und im Vordergrund liegt, startet der Wiedergabemodus, sobald Sie die Unity Szene starten. Hier müssen Sie zunächst eine JSON-Datei auswählen, aus der Sie einen Ablauf wiederherstellen möchten. Haben Sie eine Datei ausgewählt, können Sie über die
Steuerungselemente die Stage und den Zeitpunkt auswählen, den sie wiederherstellen möchten. Es ist ebenfalls möglich, den Ablauf einer Stage in Echtzeit abzuspielen.

### Einbau externer Geräte
Das Toolkit bietet die Klasse „STKReceiveSerial“, welche es ermöglicht, Daten über eine serielle Schnittstelle von einem externen Gerät auszulesen. Hierfür legen Sie das Skript auf ein beliebiges Skript und geben den gewünschten Port und die BAUD-Rate an. In „CurrentValue“ befindet sich dann die zuletzt ausgelesene Line, welche z.B. über ein Tracker-Event aufgezeichnet werden kann.

### Datenverarbeitung und Visualisierungen
Das Toolkit erstellt automatisch JSON-Dateien, welche alle aufgezeichneten Informationen enthalten. Diese können in verschiedener Statistiksoftware verwendet werden, um die Daten zu analysieren. Im Toolkit enthalten sind einige Funktionen, die die Weiterverarbeitung der Daten mit der Statistiksprache R vereinfachen sollen. Das R-Projekt befindet sich im Verzeichnis "vrstk\RTools\".
Mit Hilfe der Funktionen

**SaveFilesToWorkspace** lässt sich ein ganzer Ordner von JSON-Dateien in R
importieren. Die Dateien werden zu einer Listenvariable konvertiert. Beispielhaft
befinden sich außerdem zwei Visualisierungsskripte im Ordner.

**CreatePositionalHeatmap** erstellt eine Top-Down Heatmap von einem getrackten
Objekt, welche anzeigt, an welchen Positionen sich ein Objekt wie lange in der Szene
befunden hat. 

![Img__4_](https://user-images.githubusercontent.com/59827885/125066207-03a75d80-e0b3-11eb-801e-f8cc8fc53abc.png)
*Beispiel einer generierten Heatmap*

**CreateObjectLookGraph** verwendet Daten des „ObjectLook“ Standardevents, um
ein Balkendiagramm zu erstellen, welches zeigt, wie lange welche Objekte vom
Probanden betrachtet wurden.

![Img__5_](https://user-images.githubusercontent.com/59827885/125066224-086c1180-e0b3-11eb-9af8-246cd6967e49.png)
*Beispiel eines generierten Balkendiagramms*


### Eye Tracking
Das VRSTK unterstützt das Eye-Tracking der HTC Vive Pro Eye. Diese benötigt die SRanipal Runtime (siehe Abschnitt "Installation") sowie eine individuelle Kalibrierung über das Hauptmenü des HMD (SteamVR Dashboard)

Anleitung für die Kalibrierung:
https://developer.vive.com/us/support/sdk/category_howto/how-to-calibrate-eye-tracking.html

Die Projekt-Option "QueriesHitBackfaces" muss wie im folgenden Bild dargestellt, aktiviert werden.
![QueriesHitBackfacesOption](https://user-images.githubusercontent.com/59827885/125066246-0efa8900-e0b3-11eb-928d-a2fbd8abdcaf.png)


#### CameraRigEye-Prefab
Das CameraRigEye-Prefab beinhaltet sowohl das Eye-Tracking als auch das Tracking der angeschauten Objekte. 

![CameraRigPrefab_CameraTracking](https://user-images.githubusercontent.com/59827885/125066292-1faaff00-e0b3-11eb-9833-ee610f98b92f.png)

![CameraRigPrefab_OptionEnableObjectTracking](https://user-images.githubusercontent.com/59827885/125066314-246fb300-e0b3-11eb-8aa8-eb21c3df9cd8.png)

Weitere Optionen sind:
- LookEvent
    - EyeLookAtObject-Event für das senden der getrackten Informationen (wird für Collider- und Objekt-Tracking verwendet).
- Debug Eye Tracking Position
    - Zeichnet eine Geradensegment in die Richtung, in die ein Auge gerade schaut.
- Eye Hitpoint
    - Gibt die aktuelle Position vom getracktem Blick, der mittels eines Strahls einen Collider berührt. Ist also die Berührungspositions eines Colliders. 
- Eye Direction
    - Die getrackte Blickrichtung eines Auges
- Valid Tracking
    - Gibt an ob ein Tracking funktioniert hat "true" oder "false".


#### Gaze Tracking
Für HMDs ohne Eye-Tracking Funktionalität ist eine Gaze Tracking Option vorhanden. Dazu muss auf dem Object CameraTracker das Skript "STKLookDirection" aktiviert werden


#### Hinweis: Installation/SDK/Plugin

- Die Installation und das SDK von SRanipal werden im Unterverzeichnis ".\\Software\\EyeTrackingExtension\\" hinterlegt
    - ".\\" Root-Verzeichnis vom VRSTK-Repository
    - Version 1.3.1.0
    - URL zum Download (HTC Account muss vorhanden sein): https://hub.vive.com/en-US/download
- Das JSON.Net Plug in wurde im Unterverzeichnis ".\\Assets\\SteamVR\\Input\\Plugins\\JSON.NET\\" hinterlegt
    - FileVersion 9.0.1.19813
    - Dies wurde mittels folgender Anleitungen durchgeführt
        - URL-Fehlerbehebung: https://stackoverflow.com/questions/38084186/json-net-in-unity-throwing-the-type-or-namespace-newtonsoft-could-not-be-foun
        - URL-Datei: https://github.com/ValveSoftware/steamvr_unity_plugin/tree/master/Assets/SteamVR/Input/Plugins/JSON.NET
