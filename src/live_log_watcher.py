import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

class LogHandler(FileSystemEventHandler):

    def on_modified(self, event):
        if event.src_path.endswith("app.log"):
            print("New log entries detected.")
            

if __name__ == "__main__":
    log_dir = Path("data/live_logs")
    observer = Observer()
    observer.schedule(LogHandler(), path=str(log_dir), recursive=False)
    observer.start()

    print(f"Watching for log changes in {log_dir}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()