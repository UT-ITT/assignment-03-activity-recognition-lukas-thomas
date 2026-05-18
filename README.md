[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CjRQqtHi)

# 1. Gathering Training Data
## gather_data.py
Beim starten von gather_data.py werden die Eingaben für Name, Activity und Nummer der Aufnahme als User-Input abgefragt.\
Dann beginnt die  Aufnahme der DIPPID-Device Accelerometer und Gyroscope Daten mit Zeitstempel für 10 Sekunden.\
Das resampling auf 100Hz findet dann schon innerhalb unseres Programms statt (aus Christoph's code).

Die Aufgenommen Daten werden dann unter\
File Name: your_name-activity-number.csv\
 mit den
Columns: id, timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z gespeichert.

# 2. Activity Recognition
## fitness_trainer.py
### Classifier training
Beim starten wird zuerst ein Classifier mit den Funktionen aus activity_recognizer trainiert, dass dauert ein wenig da zuerst alle Datensätze verarbeitet werden, müssen bis das Training beginnt. Dann wird die Accuracy auf dem Testset geprinted. Man kann schon vorher die DIPPID App senden lassen.

### App Tutorial
Es wird dann eine Aktivität angezeigt (mit Animation aus den Images) die man dann für 10 Sekunden mit dem DIPPID Device ausführen soll.\
Pro Sekunde in der man es korrekt macht wird hochgezählt und ab 10 korrekten Sekunden eine neue auszuführende Aktivität zufällig gewählt, die man dann wieder von 0 Sekunden neu 10 Sekunden lang ausführen soll.\
Wird eine falsche Aktivität als die gezeigt erkannt, steht im Fenster "keep going!" und man weiß, man führt die Aktivität nicht korrekt aus.

Mit Taste Q kann man die App schließen.

### prediction
Die Sensordaten werden in 100Hz gelesen und füllen einen databuffer der genau 100 Zeilen platz hat, für je eine Sekunde Aufnahme, sodass wir immer auf eine Sekunde Daten predicten können.

Die predict Funktion wird einmal pro Sekunde ausgeführt und extrahiert features aus dem databuffer und predicted dann mit dem vorher trainierten Classifier auf den Features. D.h. man erhält eine prediction pro Sekunde.

## activity_recognizer.py
### Modell und split
Als Modell haben wir eine SVM mit rbf Kernel genommen. Die Datensätze in Train und Testsplit (80/20) aufgeteilt und dafür einen Groupsplit(nach Personen) statt einem random split genommen, weil beim random split die Datensätze einer Person sowohl im Trainings- als auch im Testset vorkommen können und wir damit einen bias haben, weil das Modell diese Daten schon "gesehen" hat.

### ausführen
Wenn man activity_recognizer alleine ausführt, wird ein classifier trainiert und darunter getestet:
- Einmal auf einen gewählten Split Seed getestet. Mit Confusion Matrix um einzelne Predictions zu zeigen.
- Einmal über 10 gewählte Seeds getestet und einen Durchschnitt berechnet.
- Und einmal je eine Person als Testset (LOGO) getestet und über alle Personen den Durchschnitt berechnet.

### Mit Windowing:

--- Results for Seed 1000 ---
Accuracy: 0.8301

Evaluating over 10 seeds...
Mean Accuracy: 0.9119

Overall LOGO Mean Accuracy: 0.8978

### Ohne Windowing:

--- Results for Seed 1000 ---
Accuracy: 0.9318

Evaluating over 10 seeds...
Mean Accuracy: 0.9665

Overall LOGO Mean Accuracy: 0.9719

### Testen auf neuen Daten (Für Tutoren):
Um die Accuracy unseres classifiers auf neuen Daten zu testen:\
Im activity_recognizer.py in der main function:
- test auf True setzen.
- den eigenen Datensatz im ordner test_data speichern
- activity-recognizer.py alleine ausführen.

## preprocessing.py
Im preprocessing werden die csv files aus dem data/ Ordner eingelesen, in ein dataframe gepackt und für jedes Dataset 1 Sekunden Fenster mit 50% Überlappung erstellt, daraus dann die Features extrahiert, auf denen unser Modell trainiert wird.\
Das Windowing machen wir weil wir in der App später jede Sekunde einmal die Aktivität predicten wollen und deshalb unser Modell auf Sekunden Aufnahmen trainieren müssen.

Wir haben uns für feature extraction mit statistischen und frequenzbasierten Features entschieden weil sie mehr über die Merkmale und Unterschiede der Aktivitäten zeigen womit unser Modell die Aktivitäten besser lernen kann als von reinen Sensordaten.

Man kann die Features aus den Datensätzen in einem features.csv file speichern indem man preprocessing.py alleine ausführt.

Wir haben die Datensätze ohne Gyroscope Daten ganz weggelassen, sowie die von Sam, weil wir denken dass er seine Daten falsch gelabelt hat. Und nur einen Jonas weil die Files falsch gelabelt waren.
