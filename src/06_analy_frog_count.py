import os
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_labels(label_folders):
    """
    Lädt Annotationen aus .Table.1.selections.txt und ordnet sie den Dateien zu.

    Args:
        label_folders (str or list of str): Pfad bzw. Pfade zu den Label-Textdateien.

    Returns:
        tuple:
         - dict: { "basename": [(start_sec, end_sec), ...] } mit allen Intervallen je Datei,
         - dict: Anzahl der Rufe pro Tag,
         - dict: Anzahl der Rufe pro (Datum, Stunde)
    """
    if isinstance(label_folders, str):
        label_folders = [label_folders]

    labels = {}
    call_counts_by_date = defaultdict(int)
    call_counts_by_date_hour = defaultdict(int)

    for lf in label_folders:
        for txt_file in os.listdir(lf):
            if not txt_file.endswith(".Table.1.selections.txt"):
                continue

            label_path = os.path.join(lf, txt_file)
            basename = txt_file.replace(".Table.1.selections.txt", "")

            # Extrahiere Datum und Uhrzeit aus dem Dateinamen (Format: _YYYYMMDD_HHMMSS)
            parts = basename.split("_")
            if len(parts) < 2:
                continue
            date_part = parts[-2]
            time_part = parts[-1]
            if len(date_part) == 8 and len(time_part) == 6:
                try:
                    file_datetime = datetime.datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                except ValueError:
                    continue
            else:
                continue

            file_date = file_datetime.date()

            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                time_intervals = []
                for line in lines[1:]:  # Header überspringen
                    parts_line = line.strip().split("\t")
                    if len(parts_line) < 5:
                        continue

                    start_time = float(parts_line[3])  # "Begin Time (s)"
                    end_time = float(parts_line[4])    # "End Time (s)"
                    time_intervals.append((start_time, end_time))

                if time_intervals:
                    labels[basename] = time_intervals
                    call_counts_by_date[file_date] += len(time_intervals)
                    call_counts_by_date_hour[(file_date, file_datetime.hour)] += len(time_intervals)
                    print(f"[INFO] Labels für {basename} geladen: {len(time_intervals)} Rufe")
            except Exception as e:
                print(f"[ERROR] Fehler beim Laden von {label_path}: {e}")

    return labels, call_counts_by_date, call_counts_by_date_hour


if __name__ == "__main__":
    # Pfade zu den Label-Dateien
    label_folders = ["../data/Inference_Results/vall_old"]

    # Lade Label-Informationen
    labels_dict, call_counts_by_date, call_counts_by_date_hour = load_labels(label_folders)

    # Speichere das Ergebnis in eine CSV-Datei (Tagesaggregat)
    out_csv = "../data/label_call_counts.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Datum", "Anzahl Rufe"])
        for date, count in sorted(call_counts_by_date.items()):
            writer.writerow([date.strftime("%Y-%m-%d"), count])

    print(f"[INFO] CSV-Datei mit Ruf-Anzahlen gespeichert unter: {out_csv}")

    # Filter nur für September 2019 bis März 2020 (ohne künstliche Null-Werte)
    filtered_dates = []
    filtered_counts = []

    for date, count in sorted(call_counts_by_date.items()):
        if datetime.date(2019, 9, 1) <= date <= datetime.date(2020, 1, 15):
            filtered_dates.append(date)
            filtered_counts.append(count)

    if not filtered_dates:
        print("[WARN] Keine Daten für den angegebenen Zeitraum vorhanden.")
    else:
        # Plot 1: Tages-Zeitreihe mit gleitendem Durchschnitt
        window_size = min(7, len(filtered_counts))  # Falls weniger als 7 Werte vorhanden sind
        smoothed_counts = np.convolve(filtered_counts, np.ones(window_size) / window_size, mode='same')

        # X-Achsen-Ticks (Monatsnamen, jeweils der 1. Tag)
        xticks_labels = [date.strftime('%b') for date in filtered_dates if date.day == 1]
        xticks_positions = [date for date in filtered_dates if date.day == 1]

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_dates, filtered_counts, marker='o', linestyle='-', color='blue', label='Daily Calls')
        plt.plot(filtered_dates, smoothed_counts, linestyle='-', color='red', label='Moving Average (7 days)')
        plt.xlabel("Months (September 2019 - March 2020)")
        plt.ylabel("Number of Calls")
        plt.title("Distribution of Calls from September 2019 to March 2020")
        plt.xticks(ticks=xticks_positions, labels=xticks_labels, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot 2: Heatmap – Aufteilung der Rufe nach Stunde und Tag
        # Es werden nur die Stunden berücksichtigt, in denen typischerweise Rufe auftreten:
        # von 0 bis 7 und von 16 bis 23.
        selected_hours = list(range(0, 8)) + list(range(16, 24))

        start_date = datetime.date(2019, 9, 1)
        end_date = datetime.date(2020, 1, 15)
        num_days = (end_date - start_date).days + 1
        dates_range = [start_date + datetime.timedelta(days=i) for i in range(num_days)]

        # Erstelle ein Array: Zeilen = ausgewählte Stunden, Spalten = Tage
        heatmap_data = np.zeros((len(selected_hours), num_days))
        for j, d in enumerate(dates_range):
            for i, hour in enumerate(selected_hours):
                heatmap_data[i, j] = call_counts_by_date_hour.get((d, hour), 0)

        plt.figure(figsize=(15, 6))
        plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Number of Calls')
        plt.xlabel("Date")
        plt.ylabel("Hour of Day")
        # X-Achsen-Beschriftung: Wähle ca. 8 gleichmäßig verteilte Tage
        tick_positions = np.linspace(0, num_days - 1, num=8, dtype=int)
        tick_labels = [dates_range[pos].strftime("%Y-%m-%d") for pos in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45)
        # Y-Achse: Zeige die tatsächlich ausgewählten Stunden
        plt.yticks(ticks=np.arange(len(selected_hours)), labels=selected_hours)
        plt.title("Heatmap of Calls by Hour (Sept 2019 - Mar 2020)\n(Only hours 0-7 & 16-23)")
        plt.tight_layout()
        plt.show()